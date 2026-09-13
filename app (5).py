"""
PharmaFlow v4.3 — Sistema Inteligente de Gestion de Compras Farmaceuticas
Streamlit · Motor Hibrido: Heuristico + ML + Ensemble

Requisitos:
  pip install streamlit pandas numpy plotly openpyxl xlsxwriter python-dateutil scikit-learn xgboost joblib
Ejecucion:
  streamlit run app.py

Novedades v4.3 (sobre v4.2):
  - Features ML mejoradas: cobro pensiones (zona cobro + paga extra), categoria ATC,
    temperatura mensual, ratio stock/venta
  - Motor Ensemble: pondera ML + Heuristico por confianza por producto
  - Tabla maestra ATC opcional (Excel CN -> ATC)
  - Temperatura historica opcional (CSV)
  - Zona climatica en perfil de farmacia
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from fpdf import FPDF
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

try:
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

APP_NAME = "PharmaFlow"
APP_ICON = "\U0001f48a"
VERSION = "4.3"

COL_CN = "Codigo_Nacional"
COL_NOMBRE = "Nombre"
COL_STOCK = "Stock"
COL_PVL = "PVL"
COL_LAB = "Laboratorio"
COL_VENTAS = "Ventas"
COL_FECHA = "Fecha"
COL_MOLECULA = "Molecula"

BASE_DIR = Path(__file__).parent / "farmacias"

COLORS = {
    "primary": "#0066FF", "secondary": "#00C49A", "danger": "#FF4B4B",
    "warning": "#FFA726", "bg_card": "#F8F9FC", "text": "#1E293B",
    "muted": "#64748B", "success": "#10B981",
}

Z_SCORES = {95: 1.645, 99: 2.326}

HEALTH_SCORE_MESES_DEFAULT = 2.0


# ===========================================================================
# CSS + RENDERING UTILS
# ===========================================================================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .pharma-header {
        background: linear-gradient(135deg, #0066FF 0%, #00C49A 100%);
        padding: 1.5rem 2rem; border-radius: 16px; color: white;
        margin-bottom: 1.5rem; display: flex; align-items: center;
        justify-content: space-between;
    }
    .pharma-header h1 { margin:0; font-size:1.8rem; font-weight:700; }
    .pharma-header .version {
        background: rgba(255,255,255,0.2); padding: 4px 12px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
    .kpi-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 1.25rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .kpi-label { font-size:0.8rem; color:#64748B; font-weight:500;
        text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }
    .kpi-value { font-size:1.8rem; font-weight:700; color:#1E293B; line-height:1.2; }
    .kpi-delta { font-size:0.85rem; font-weight:500; margin-top:4px; }
    .kpi-delta.positive { color:#10B981; }
    .kpi-delta.negative { color:#FF4B4B; }
    .upload-zone { border:2px dashed #CBD5E1; border-radius:12px; padding:2rem;
        text-align:center; background:#F8FAFC; transition:border-color 0.2s; }
    .upload-zone:hover { border-color:#0066FF; }
    .budget-bar-container { background:#E2E8F0; border-radius:8px;
        overflow:hidden; height:24px; position:relative; }
    .budget-bar-fill { height:100%; border-radius:8px; transition:width 0.4s ease;
        display:flex; align-items:center; justify-content:center;
        font-size:0.75rem; font-weight:600; color:white; }
    .dataframe { font-size: 0.85rem !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0;
        padding:10px 20px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)


def render_header(farmacia_nombre=""):
    subtitulo = f"Farmacia: {farmacia_nombre}" if farmacia_nombre else "Sistema Inteligente de Gestion de Compras"
    st.markdown(f"""
    <div class="pharma-header">
        <div><h1>{APP_ICON} {APP_NAME}</h1>
        <span style="opacity:0.85;font-size:0.9rem;">{subtitulo}</span></div>
        <span class="version">v{VERSION}</span>
    </div>""", unsafe_allow_html=True)


def render_kpi(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_positive else "negative"
        icon = "\u2191" if delta_positive else "\u2193"
        delta_html = f'<div class="kpi-delta {cls}">{icon} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def format_eur(value):
    if abs(value) >= 1000:
        return f"{value:,.2f} \u20ac".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:.2f} \u20ac".replace(".", ",")

def safe_div(a, b, default=0):
    """Division segura evitando ZeroDivisionError."""
    try:
        return float(a) / float(b) if round(float(b), 4) != 0 else default
    except (ValueError, TypeError):
        return default

def heuristica_tipo_producto(cn, molecula=""):
    """Clasifica como Medicamento o Parafarmacia evaluando C.N., EAN y la familia."""
    mol_str = str(molecula).strip().lower()
    
    # Prioridad 1: Si la familia o molécula grita "Parafarmacia" explícitamente
    if any(k in mol_str for k in ["parafarmacia", "cosmetica", "dietetica", "higiene", "ortopedia"]):
        return "Parafarmacia"
        
    cn_str = str(cn).strip()
    if not cn_str or cn_str.lower() in ["nan", "none"]:
        return "Parafarmacia"
        
    # Lecturas de un código QR (DataMatrix SEVeM / GTIN-14): 
    # El identificador de medicamentos españoles siempre contiene la base 847000 en el interior.
    if "847000" in cn_str:
        return "Medicamento"
        
    # Códigos de Barras largos (EAN-12, EAN-13, EAN-14) que NO contienen el patrón 847000
    if len(cn_str) >= 12:
        return "Parafarmacia"
        
    # Códigos Nacionales cortos tradicionales (6 o 7 dígitos)
    # Los medicamentos de prescripción y OTC suelen arrancar en 6, 7, 8 o 9.
    # Los productos de higiene, dietética (5), ortopedia (4) y parafarmacia (1,2,3) arrancan más bajo.
    if cn_str[0] in ['6', '7', '8', '9']:
        return "Medicamento"
        
    return "Parafarmacia"

def asignar_grupo_surtido(nombre, molecula, reglas_list):
    """
    Si una regla separada por comas (ej. 'omeprazol,20,28') hace match total 
    dentro del Nombre del producto, se clasifica bajo esa regla granulada.
    """
    nom_lower = str(nombre).lower()
    for regla in reglas_list:
        if not regla: continue
        partes = [p.strip() for p in regla.split(",") if p.strip()]
        if partes and all(p in nom_lower for p in partes):
            return regla
            
    mol_str = str(molecula).strip()
    # Evitamos cajones de sastre inútiles que falseen el panel (mezclando geles con tiritas)
    if pd.isna(molecula) or mol_str.lower() in ["parafarmacia", "otros", "varios", "desconocido", "", "-", "cosmetica", "higiene", "dietetica", "infantil"]:
        return str(nombre).strip()
        
    return mol_str

# ===========================================================================
# MULTI-FARMACIA + PERSISTENCIA JSON
# ===========================================================================
def obtener_farmacias_disponibles():
    if not BASE_DIR.exists():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])

def crear_farmacia(nombre):
    nombre_limpio = nombre.strip().replace(" ", "_").replace("/", "-")
    ruta = BASE_DIR / nombre_limpio
    ruta.mkdir(parents=True, exist_ok=True)
    return nombre_limpio

def ruta_farmacia_activa():
    nombre = st.session_state.get("farmacia_activa")
    if nombre:
        return BASE_DIR / nombre
    return None

def cargar_json_farmacia(nombre_archivo, default=None):
    ruta = ruta_farmacia_activa()
    if ruta is None:
        return default if default is not None else {}
    filepath = ruta / nombre_archivo
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default if default is not None else {}
    return default if default is not None else {}

def guardar_json_farmacia(nombre_archivo, data):
    ruta = ruta_farmacia_activa()
    if ruta is None:
        return
    filepath = ruta / nombre_archivo
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def guardar_dataframe_farmacia(nombre_archivo, df):
    """Guarda un DataFrame como Parquet en la carpeta de la farmacia activa."""
    ruta = ruta_farmacia_activa()
    if ruta is None or df is None or df.empty:
        return
    filepath = ruta / nombre_archivo
    try:
        df.to_parquet(filepath, index=False, engine="pyarrow")
    except ImportError:
        # Fallback a CSV si pyarrow no esta instalado
        filepath_csv = filepath.with_suffix(".csv")
        df.to_csv(filepath_csv, index=False, encoding="utf-8")
    except Exception:
        pass


def cargar_dataframe_farmacia(nombre_archivo):
    """Carga un DataFrame desde Parquet. Devuelve None si no existe."""
    ruta = ruta_farmacia_activa()
    if ruta is None:
        return None
    filepath = ruta / nombre_archivo
    if filepath.exists():
        try:
            return pd.read_parquet(filepath, engine="pyarrow")
        except ImportError:
            # Fallback a CSV
            filepath_csv = filepath.with_suffix(".csv")
            if filepath_csv.exists():
                return pd.read_csv(filepath_csv, encoding="utf-8")
        except Exception:
            pass
    # Fallback: buscar CSV si Parquet no existe
    filepath_csv = filepath.with_suffix(".csv")
    if filepath_csv.exists():
        try:
            return pd.read_csv(filepath_csv, encoding="utf-8")
        except Exception:
            pass
    return None


def _autocargar_datos_farmacia():
    """Carga automaticamente inventario, historico y ofertas desde disco al iniciar sesion."""
    cargados = []
    if "inventario" not in st.session_state:
        df = cargar_dataframe_farmacia("inventario.parquet")
        if df is not None and not df.empty:
            st.session_state["inventario"] = df
            cargados.append(f"Inventario ({len(df)} productos)")
    if "historico" not in st.session_state:
        df = cargar_dataframe_farmacia("historico.parquet")
        if df is not None and not df.empty:
            st.session_state["historico"] = df
            cargados.append(f"Historico ({len(df)} registros)")
    if "ofertas_normalizadas" not in st.session_state:
        df = cargar_dataframe_farmacia("ofertas_normalizadas.parquet")
        if df is not None and not df.empty:
            st.session_state["ofertas_normalizadas"] = df
            cargados.append(f"Ofertas ({len(df)} items)")
    return cargados


# ===========================================================================
# META-ANALISIS + REGLAS + PROTEGIDOS
# ===========================================================================
def registrar_snapshot_kpi(health_score, n_zombies, valor_zombie, n_roturas, ahorro_pedido=0, coste_pedido=0):
    historico = cargar_json_farmacia("historico_kpi.json", default=[])
    historico.append({
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "health_score": health_score, "n_zombies": n_zombies,
        "valor_zombie": round(valor_zombie, 2), "n_roturas": n_roturas,
        "ahorro_pedido": round(ahorro_pedido, 2), "coste_pedido": round(coste_pedido, 2),
    })
    guardar_json_farmacia("historico_kpi.json", historico)
    return historico

def obtener_historico_kpi():
    historico = cargar_json_farmacia("historico_kpi.json", default=[])
    if not historico:
        return pd.DataFrame()
    df = pd.DataFrame(historico)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

def cargar_reglas_surtido():
    return cargar_json_farmacia("reglas_surtido.json", default=[])
def guardar_reglas_surtido(reglas):
    guardar_json_farmacia("reglas_surtido.json", reglas)
def cargar_productos_protegidos():
    return cargar_json_farmacia("productos_protegidos.json", default=[])
def guardar_productos_protegidos(protegidos):
    guardar_json_farmacia("productos_protegidos.json", protegidos)


# ===========================================================================
# SNAPSHOT DE AUDITORIA (evolucion temporal del inventario)
# ===========================================================================
def registrar_snapshot_auditoria(df_inventario, df_ventas_media):
    """Guarda resumen ligero del inventario para tracking temporal."""
    df = df_inventario.merge(df_ventas_media, on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)

    # Zombies
    zombies = df[(df[COL_STOCK] > 0) & (df["Venta_Media_Mensual"] == 0)]
    valor_zombie_total = round(float((zombies[COL_STOCK] * zombies.get(COL_PVL, pd.Series(0, index=zombies.index))).fillna(0).sum()), 2)
    zombie_list = []
    for _, r in zombies.head(30).iterrows():
        val = float(r[COL_STOCK]) * float(r.get(COL_PVL, 0))
        zombie_list.append({"cn": r[COL_CN], "nombre": str(r.get(COL_NOMBRE, ""))[:50], "valor": round(val, 2)})

    # Roturas
    roturas = df[(df[COL_STOCK] == 0) & (df["Venta_Media_Mensual"] > 0)]
    rotura_list = [{"cn": r[COL_CN], "nombre": str(r.get(COL_NOMBRE, ""))[:50],
                    "venta_media": round(float(r["Venta_Media_Mensual"]), 1)}
                   for _, r in roturas.head(20).iterrows()]

    # Sobrestock top 20
    df["Stock_Ideal"] = df["Venta_Media_Mensual"] * HEALTH_SCORE_MESES_DEFAULT
    df["Exceso"] = (df[COL_STOCK] - df["Stock_Ideal"]).clip(lower=0)
    df["Valor_Exceso"] = (df["Exceso"] * df.get(COL_PVL, pd.Series(0))).fillna(0)
    sobr = df[df["Exceso"] > 0].nlargest(20, "Valor_Exceso")
    sobrestock_list = [{"cn": r[COL_CN], "nombre": str(r.get(COL_NOMBRE, ""))[:50],
                        "exceso": int(r["Exceso"]), "valor": round(float(r["Valor_Exceso"]), 2)}
                       for _, r in sobr.iterrows()]

    snapshot = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_productos": len(df_inventario),
        "n_zombies": len(zombies),
        "valor_zombie": valor_zombie_total,
        "n_roturas": len(roturas),
        "valor_sobrestock": round(float(df["Valor_Exceso"].sum()), 2),
        "zombies": zombie_list,
        "roturas": rotura_list,
        "sobrestock": sobrestock_list,
    }

    historico = cargar_json_farmacia("historico_auditorias.json", default=[])
    historico.append(snapshot)
    # Mantener solo ultimos 24 snapshots
    if len(historico) > 24:
        historico = historico[-24:]
    guardar_json_farmacia("historico_auditorias.json", historico)
    return snapshot


def obtener_historico_auditorias():
    return cargar_json_farmacia("historico_auditorias.json", default=[])


# ===========================================================================
# PERFIL DE FARMACIA (3 niveles epidemiologicos)
# ===========================================================================
def cargar_perfil_farmacia():
    return cargar_json_farmacia("perfil_farmacia.json", default={
        "centro_salud": False, "residencia": False, "colegio": False,
        "zona_turistica": False, "zona_rural": False,
        "epi_gripe": 0, "epi_alergias": 0, "epi_covid": 0,
    })

def guardar_perfil_farmacia(perfil):
    guardar_json_farmacia("perfil_farmacia.json", perfil)


# ===========================================================================
# CALENDARIO: 3 CAPAS (Horario base + Festivos + Excepciones)
# ===========================================================================
def cargar_calendario_farmacia():
    return cargar_json_farmacia("calendario_farmacia.json", default={
        "horario": {
            "lunes": {"abre": True, "apertura": "09:30", "cierre": "21:00"},
            "martes": {"abre": True, "apertura": "09:30", "cierre": "21:00"},
            "miercoles": {"abre": True, "apertura": "09:30", "cierre": "21:00"},
            "jueves": {"abre": True, "apertura": "09:30", "cierre": "21:00"},
            "viernes": {"abre": True, "apertura": "09:30", "cierre": "21:00"},
            "sabado": {"abre": True, "apertura": "10:00", "cierre": "14:00"},
            "domingo": {"abre": False, "apertura": "00:00", "cierre": "00:00"},
        },
        "festivos": [],
        "excepciones_abiertas": [],
    })

def guardar_calendario_farmacia(calendario):
    guardar_json_farmacia("calendario_farmacia.json", calendario)


def _dia_abre(d, horario, festivos_str, excepciones_str):
    """Determina si un dia concreto la farmacia abre y devuelve (abre, config_dia)."""
    dias_semana_es = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    d_str = d.strftime("%d/%m/%Y")
    dia_semana = dias_semana_es[d.weekday()]
    config_dia = horario.get(dia_semana, {"abre": False})
    es_festivo = d_str in festivos_str
    es_excepcion = d_str in excepciones_str
    if es_excepcion:
        return True, config_dia
    elif es_festivo:
        return False, config_dia
    else:
        return config_dia.get("abre", False), config_dia


def calcular_horas_mes(anio, mes, calendario):
    """Calcula horas totales de apertura en un mes dado."""
    horario = calendario.get("horario", {})
    festivos_str = set(calendario.get("festivos", []))
    excepciones_str = set(calendario.get("excepciones_abiertas", []))
    total_dias = monthrange(anio, mes)[1]
    horas_total = 0.0
    for dia in range(1, total_dias + 1):
        try:
            d = date(anio, mes, dia)
        except ValueError:
            continue
        abre, config_dia = _dia_abre(d, horario, festivos_str, excepciones_str)
        if abre:
            try:
                h_ap = config_dia.get("apertura", "09:30").split(":")
                h_ci = config_dia.get("cierre", "21:00").split(":")
                horas = (int(h_ci[0]) + int(h_ci[1]) / 60) - (int(h_ap[0]) + int(h_ap[1]) / 60)
                horas_total += max(0, horas)
            except (ValueError, IndexError):
                horas_total += 8.0
    return round(horas_total, 1)


def calcular_dias_abiertos_mes(anio, mes, calendario):
    """Calcula dias abiertos en un mes dado."""
    horario = calendario.get("horario", {})
    festivos_str = set(calendario.get("festivos", []))
    excepciones_str = set(calendario.get("excepciones_abiertas", []))
    total_dias = monthrange(anio, mes)[1]
    abiertos = 0
    for dia in range(1, total_dias + 1):
        try:
            d = date(anio, mes, dia)
        except ValueError:
            continue
        abre, _ = _dia_abre(d, horario, festivos_str, excepciones_str)
        if abre:
            abiertos += 1
    return abiertos


# ===========================================================================
# REGISTRO HISTORICO DE PROMOCIONES
# ===========================================================================
def registrar_promociones_pedido(df_pedido):
    """Guarda registro plano de productos con oferta en el pedido actual."""
    if df_pedido is None or df_pedido.empty:
        return
    if "Descuento_Aplicado" not in df_pedido.columns:
        return
    df_promo = df_pedido[df_pedido["Descuento_Aplicado"] > 0].copy()
    if df_promo.empty:
        return

    registros = cargar_json_farmacia("historico_promociones.json", default=[])
    fecha = datetime.now().strftime("%Y-%m-%d")
    for _, row in df_promo.iterrows():
        registros.append({
            "fecha": fecha,
            "codigo_nacional": row.get(COL_CN, ""),
            "molecula": row.get(COL_MOLECULA, ""),
            "laboratorio": row.get(COL_LAB, ""),
            "descuento": round(float(row.get("Descuento_Aplicado", 0)), 4),
            "tier": row.get("Tier_Aplicado", ""),
        })
    guardar_json_farmacia("historico_promociones.json", registros)


def obtener_cns_con_promo_historica():
    registros = cargar_json_farmacia("historico_promociones.json", default=[])
    return {r["codigo_nacional"] for r in registros if r.get("codigo_nacional")}


# ===========================================================================
# VALIDACION + RENOMBRADO
# ===========================================================================
def validar_y_renombrar_columnas(df, columnas_requeridas):
    df = df.copy()
    informe = {}
    rename_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for col_esperada in columnas_requeridas:
        col_lower = col_esperada.lower().strip()
        encontrada = None
        if col_lower in cols_lower:
            encontrada = cols_lower[col_lower]
        else:
            for df_col_lower, df_col_original in cols_lower.items():
                if col_lower in df_col_lower or df_col_lower in col_lower:
                    encontrada = df_col_original
                    break
        if encontrada and encontrada != col_esperada:
            rename_map[encontrada] = col_esperada
        informe[col_esperada] = encontrada
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, informe


# ===========================================================================
# NORMALIZACION DE OFERTAS — Tiers Dinamicos
# ===========================================================================
def normalizar_ofertas_dinamico(df_raw, mapping_nombre, mappings_tiers):
    df = df_raw.copy()
    resultados = []
    for _, row in df.iterrows():
        nombre = str(row.get(mapping_nombre, "")).strip()
        if not nombre:
            continue
        tiers = []
        for mt in mappings_tiers:
            col_cant = mt.get("col_cantidad")
            col_dto = mt.get("col_descuento")
            if not col_cant or not col_dto:
                continue
            cant = pd.to_numeric(str(row.get(col_cant, 0)).strip(), errors="coerce")
            dto_raw = str(row.get(col_dto, 0)).replace("%", "").replace(",", ".").strip()
            dto = pd.to_numeric(dto_raw, errors="coerce")
            if pd.isna(cant) or pd.isna(dto) or cant <= 0 or dto <= 0:
                continue
            if dto > 1:
                dto = dto / 100
            tiers.append({"min": int(cant), "dto": float(dto)})
        tiers.sort(key=lambda t: t["min"])
        if tiers:
            resultados.append({"Oferta_Nombre": nombre, "Tiers": tiers})
    return pd.DataFrame(resultados)


# ===========================================================================
# IMPUTACION DE STOCKOUTS
# ===========================================================================
def imputar_stockouts(df_ventas, df_inventario):
    df = df_ventas.copy()
    if COL_STOCK not in df_inventario.columns:
        return df
    stock_map = df_inventario.set_index(COL_CN)[COL_STOCK].to_dict()
    cns_sin_stock = {cn for cn, s in stock_map.items() if s == 0}
    if not cns_sin_stock:
        return df
    for cn in cns_sin_stock:
        mask_cn = df[COL_CN] == cn
        df_cn = df[mask_cn].sort_values(COL_FECHA if COL_FECHA in df.columns else COL_CN)
        if COL_VENTAS not in df_cn.columns:
            continue
        ventas = df_cn[COL_VENTAS].values.copy()
        periodos_con_venta = ventas[ventas > 0]
        if len(periodos_con_venta) == 0:
            continue
        media_movil = periodos_con_venta[-7:].mean() if len(periodos_con_venta) >= 7 else periodos_con_venta.mean()
        for i in range(len(ventas) - 1, -1, -1):
            if ventas[i] == 0:
                ventas[i] = max(1, int(round(media_movil)))
            else:
                break
        df.loc[mask_cn, COL_VENTAS] = list(ventas)
    return df


# ===========================================================================
# FORECASTING HEURISTICO (Mirroring + Tendencia)
# ===========================================================================
def calcular_ventas_mensuales(df_ventas, meses_cobertura=2):
    df = df_ventas.copy()
    if COL_FECHA not in df.columns:
        agg = df.groupby(COL_CN)[COL_VENTAS].agg(["sum", "std"]).reset_index()
        agg.columns = [COL_CN, "total", "std_raw"]
        agg["Venta_Media_Mensual"] = agg["total"] / 12
        agg["Venta_Std_Mensual"] = agg["std_raw"].fillna(0) / np.sqrt(12)
        agg["Factor_Tendencia"] = 0.0
        return agg[[COL_CN, "Venta_Media_Mensual", "Venta_Std_Mensual", "Factor_Tendencia"]]

    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_FECHA])
    df["Anio"] = df[COL_FECHA].dt.year
    df["Mes_Cal"] = df[COL_FECHA].dt.month
    ventas_por_mes = df.groupby([COL_CN, "Anio", "Mes_Cal"])[COL_VENTAS].sum().reset_index()
    anios_unicos = sorted(ventas_por_mes["Anio"].unique())
    anio_max = anios_unicos[-1]

    cns_unicos_ft = ventas_por_mes[COL_CN].unique()
    factor_tendencia = pd.DataFrame({COL_CN: cns_unicos_ft, "Factor_Tendencia": [0.0] * len(cns_unicos_ft)})
    if len(anios_unicos) >= 2:
        ventas_anual = ventas_por_mes.groupby([COL_CN, "Anio"])[COL_VENTAS].sum().reset_index().sort_values([COL_CN, "Anio"])
        crecimientos = []
        for cn, grupo in ventas_anual.groupby(COL_CN):
            if len(grupo) < 2:
                crecimientos.append({COL_CN: cn, "Factor_Tendencia": 0.0}); continue
            vals = grupo[COL_VENTAS].values
            tasas = [(vals[i]/vals[i-1] - 1) for i in range(1, len(vals)) if vals[i-1] > 0]
            if tasas:
                pesos = list(range(1, len(tasas)+1))
                crecimientos.append({COL_CN: cn, "Factor_Tendencia": round(np.clip(np.average(tasas, weights=pesos), -0.5, 0.5), 4)})
            else:
                crecimientos.append({COL_CN: cn, "Factor_Tendencia": 0.0})
        factor_tendencia = pd.DataFrame(crecimientos)

    hoy = datetime.now()
    meses_futuro = [(hoy + relativedelta(months=m)).month for m in range(1, meses_cobertura+1)]
    mirror = ventas_por_mes[(ventas_por_mes["Anio"] == anio_max) & (ventas_por_mes["Mes_Cal"].isin(meses_futuro))]
    venta_mirror = mirror.groupby(COL_CN)[COL_VENTAS].mean().reset_index().rename(columns={COL_VENTAS: "Venta_Media_Mensual"}) if not mirror.empty else pd.DataFrame(columns=[COL_CN, "Venta_Media_Mensual"])
    media_global = ventas_por_mes.groupby(COL_CN)[COL_VENTAS].mean().reset_index().rename(columns={COL_VENTAS: "Venta_Media_Global"})
    std_mensual = ventas_por_mes.groupby(COL_CN)[COL_VENTAS].std().fillna(0).reset_index().rename(columns={COL_VENTAS: "Venta_Std_Mensual"})

    vm = media_global.merge(venta_mirror, on=COL_CN, how="left")
    vm["Venta_Media_Mensual"] = vm["Venta_Media_Mensual"].fillna(vm["Venta_Media_Global"])
    vm = vm.drop(columns=["Venta_Media_Global"]).merge(std_mensual, on=COL_CN, how="left").merge(factor_tendencia, on=COL_CN, how="left")
    vm["Venta_Std_Mensual"] = vm["Venta_Std_Mensual"].fillna(0)
    vm["Factor_Tendencia"] = vm["Factor_Tendencia"].fillna(0)
    vm["Venta_Media_Mensual"] = (vm["Venta_Media_Mensual"] * (1 + vm["Factor_Tendencia"])).clip(lower=0)
    return vm


# ===========================================================================
# MOTOR ML — Feature Engineering + XGBoost
# ===========================================================================
def build_features(df_ventas, df_inventario, perfil, calendario, df_ofertas_norm=None):
    df = df_ventas.copy()
    if COL_FECHA not in df.columns:
        return pd.DataFrame()

    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_FECHA])
    df["Anio"] = df[COL_FECHA].dt.year
    df["Mes"] = df[COL_FECHA].dt.month

    mensual = df.groupby([COL_CN, "Anio", "Mes"])[COL_VENTAS].sum().reset_index()

    # Base_Mirroring
    mirror = mensual.copy().rename(columns={COL_VENTAS: "Base_Mirroring"})
    mirror["Anio"] = mirror["Anio"] + 1
    mensual = mensual.merge(mirror[[COL_CN, "Anio", "Mes", "Base_Mirroring"]],
                            on=[COL_CN, "Anio", "Mes"], how="left")
    mensual["Base_Mirroring"] = mensual["Base_Mirroring"].fillna(0)

    # Growth_Factor
    ventas_anual = mensual.groupby([COL_CN, "Anio"])[COL_VENTAS].sum().reset_index()
    growth = {}
    for cn, g in ventas_anual.groupby(COL_CN):
        vals = g.sort_values("Anio")[COL_VENTAS].values
        if len(vals) >= 2 and vals[-2] > 0:
            growth[cn] = np.clip(vals[-1] / vals[-2] - 1, -0.5, 0.5)
        else:
            growth[cn] = 0.0
    mensual["Growth_Factor"] = mensual[COL_CN].map(growth).fillna(0)

    # Lag_30
    mensual["Lag_30"] = mensual.groupby(COL_CN)[COL_VENTAS].shift(1).fillna(0)

    # CAMBIO 1: Cobro pensiones mejorado (reemplaza Days_Since_Payday)
    mensual["Pct_Zona_Cobro"] = mensual.apply(
        lambda r: _pct_zona_cobro(r["Anio"], r["Mes"]), axis=1)
    mensual["Es_Paga_Extra"] = mensual["Mes"].isin([6, 12]).astype(int)

    # Horas_Abierto_Mes
    mensual["Horas_Abierto"] = mensual.apply(
        lambda r: calcular_horas_mes(int(r["Anio"]), int(r["Mes"]), calendario), axis=1)

    # Dias_Abiertos
    mensual["Dias_Abiertos"] = mensual.apply(
        lambda r: calcular_dias_abiertos_mes(int(r["Anio"]), int(r["Mes"]), calendario), axis=1)

    # Epidemiologico (3 campos separados)
    mensual["Epi_Gripe"] = perfil.get("epi_gripe", 0)
    mensual["Epi_Alergias"] = perfil.get("epi_alergias", 0)
    mensual["Epi_Covid"] = perfil.get("epi_covid", 0)

    # Perfil farmacia
    for key in ["centro_salud", "residencia", "colegio", "zona_turistica", "zona_rural"]:
        mensual[f"Perfil_{key}"] = int(perfil.get(key, False))

    # Mes_Num
    mensual["Mes_Num"] = mensual["Mes"]

    # Was_Promo
    cns_promo = obtener_cns_con_promo_historica()
    mensual["Was_Promo"] = mensual[COL_CN].isin(cns_promo).astype(int)

    # Is_Future_Promo
    cns_oferta_actual = set()
    if df_ofertas_norm is not None and not df_ofertas_norm.empty:
        cns_oferta_actual = set(df_ofertas_norm["Oferta_Nombre"].str.lower())
    if COL_MOLECULA in df_inventario.columns:
        mol_map = df_inventario.set_index(COL_CN).get(COL_MOLECULA, pd.Series(dtype=str)).to_dict()
        mensual["_mol"] = mensual[COL_CN].map(mol_map).fillna("").str.lower()
        mensual["Is_Future_Promo"] = mensual["_mol"].isin(cns_oferta_actual).astype(int)
        mensual = mensual.drop(columns=["_mol"])
    else:
        mensual["Is_Future_Promo"] = 0

    # CAMBIO 2: Categoria terapeutica ATC (target encoding con expanding mean para evitar leakage)
    atc_map = _obtener_mapa_atc(df_inventario)
    mensual["_grupo_atc"] = mensual[COL_CN].map(atc_map).fillna("OTRO")
    # Expanding mean por grupo ATC: cada fila usa solo la media de filas anteriores
    mensual = mensual.sort_values([COL_CN, "Anio", "Mes"]).reset_index(drop=True)
    global_mean_ventas = mensual[COL_VENTAS].mean()
    atc_cumsum = mensual.groupby("_grupo_atc")[COL_VENTAS].cumsum() - mensual[COL_VENTAS]
    atc_cumcount = mensual.groupby("_grupo_atc").cumcount()
    mensual["ATC_Encoded"] = np.where(atc_cumcount > 0, atc_cumsum / atc_cumcount, global_mean_ventas)
    mensual["ATC_Encoded"] = mensual["ATC_Encoded"].round(2)
    mensual = mensual.drop(columns=["_grupo_atc"])

    # CAMBIO 3: Temperatura mensual
    temp_data = st.session_state.get("temperatura_historica")
    zona = perfil.get("zona_climatica", "mediterraneo")
    mensual["Temp_Media"] = mensual.apply(
        lambda r: _obtener_temperatura(int(r["Anio"]), int(r["Mes"]), temp_data, zona), axis=1)
    # Media climatologica por mes para calcular desviacion (solo util con datos reales)
    if temp_data is not None and not temp_data.empty:
        temp_clima = mensual.groupby("Mes")["Temp_Media"].mean().to_dict()
        mensual["Temp_Desviacion"] = mensual.apply(
            lambda r: round(r["Temp_Media"] - temp_clima.get(r["Mes"], r["Temp_Media"]), 1), axis=1)
    else:
        # Sin datos externos, climatologia = constante por mes → desviacion siempre 0
        mensual["Temp_Desviacion"] = 0.0

    # CAMBIO 4: Ratio stock/venta (solo informativo para la prediccion futura)
    # En filas historicas el stock actual no existia, asi que usamos 0 para entrenamiento.
    # El valor real se inyecta solo en df_futuro al predecir.
    if COL_STOCK in df_inventario.columns:
        stock_map = df_inventario.set_index(COL_CN)[COL_STOCK].to_dict()
        vm_global = mensual.groupby(COL_CN)[COL_VENTAS].mean().to_dict()
        # Solo la ultima fila de cada producto (la mas reciente) tiene stock ratio real
        ultimo_idx = mensual.groupby(COL_CN).tail(1).index
        mensual["Stock_Ratio"] = 0.0
        for _sr_idx in ultimo_idx:
            _sr_cn = mensual.at[_sr_idx, COL_CN]
            mensual.at[_sr_idx, "Stock_Ratio"] = round(safe_div(stock_map.get(_sr_cn, 0), vm_global.get(_sr_cn, 1)), 2)
    else:
        mensual["Stock_Ratio"] = 0.0

    return mensual


# --- Helpers para nuevas features ---
def _pct_zona_cobro(anio, mes):
    """Proporcion de dias del mes en zona de cobro de pensiones (dias 25-fin)."""
    dias = monthrange(int(anio), int(mes))[1]
    dias_cobro = max(0, dias - 24)
    return round(dias_cobro / dias, 3)

def _obtener_mapa_atc(df_inventario):
    """Devuelve dict CN -> grupo ATC. Usa tabla maestra si existe, sino proxy por molecula."""
    atc_tabla = st.session_state.get("tabla_maestra_atc")
    if atc_tabla is not None and not atc_tabla.empty:
        cn_col = [c for c in atc_tabla.columns if "codigo" in c.lower() or "cn" in c.lower() or "nacional" in c.lower()]
        atc_col = [c for c in atc_tabla.columns if "atc" in c.lower() or "grupo" in c.lower()]
        if cn_col and atc_col:
            mapa = atc_tabla.set_index(cn_col[0])[atc_col[0]].to_dict()
            # Convertir a string keys
            return {str(k): str(v)[:5] for k, v in mapa.items() if pd.notna(v)}

    # Proxy: agrupar por molecula, asignar "PARA" a parafarmacia
    if COL_MOLECULA in df_inventario.columns:
        mol_map = df_inventario.set_index(COL_CN)[COL_MOLECULA].to_dict()
        grupo_map = {}
        for cn, mol in mol_map.items():
            mol_str = str(mol).strip().lower()
            if mol_str in ("parafarmacia", "", "nan"):
                grupo_map[str(cn)] = "PARA"
            else:
                grupo_map[str(cn)] = mol_str[:20]
        return grupo_map
    return {}


# Temperaturas medias mensuales por zona climatica (España, referencia AEMET)
TEMP_CLIMATICA = {
    "mediterraneo": [10.5, 11.2, 13.5, 15.8, 19.5, 24.0, 27.5, 27.2, 23.8, 18.5, 13.5, 10.8],
    "continental":  [5.5,  7.0,  10.5, 13.0, 17.0, 22.5, 26.0, 25.5, 20.5, 14.5, 9.0,  6.0],
    "atlantico":    [9.0,  9.5,  11.5, 12.5, 15.0, 18.0, 20.5, 20.8, 19.0, 15.5, 12.0, 9.8],
}

def _obtener_temperatura(anio, mes, temp_data, zona):
    """Devuelve temperatura mensual. Usa datos reales si existen, sino climatologia."""
    if temp_data is not None and not temp_data.empty:
        match = temp_data[(temp_data["Anio"] == anio) & (temp_data["Mes"] == mes)]
        if not match.empty:
            return round(float(match.iloc[0].get("Temp_Media", 0)), 1)
    # Fallback a climatologia
    zona_key = str(zona).lower().replace("á", "a").replace("é", "e")
    if zona_key not in TEMP_CLIMATICA:
        zona_key = "mediterraneo"
    return TEMP_CLIMATICA[zona_key][mes - 1]


def cold_start_proxy(df_features, df_inventario, min_meses=6):
    conteo = df_features.groupby(COL_CN).size().reset_index(name="n_meses")
    cns_nuevos = set(conteo[conteo["n_meses"] < min_meses][COL_CN])
    if not cns_nuevos or COL_MOLECULA not in df_inventario.columns:
        return df_features
    mol_map = df_inventario.set_index(COL_CN).get(COL_MOLECULA, pd.Series(dtype=str)).to_dict()
    df_features["_mol_proxy"] = df_features[COL_CN].map(mol_map)
    df_maduros = df_features[~df_features[COL_CN].isin(cns_nuevos)]
    if df_maduros.empty:
        df_features = df_features.drop(columns=["_mol_proxy"])
        return df_features
    media_mol = df_maduros.groupby("_mol_proxy")[COL_VENTAS].mean().to_dict()
    mask_nuevo = df_features[COL_CN].isin(cns_nuevos)
    df_features.loc[mask_nuevo, COL_VENTAS] = df_features.loc[mask_nuevo, "_mol_proxy"].map(media_mol).fillna(
        df_features.loc[mask_nuevo, COL_VENTAS])
    df_features = df_features.drop(columns=["_mol_proxy"])
    return df_features


FEATURE_COLS = [
    "Base_Mirroring", "Growth_Factor", "Lag_30",
    "Pct_Zona_Cobro", "Es_Paga_Extra",
    "Horas_Abierto", "Dias_Abiertos",
    "Epi_Gripe", "Epi_Alergias", "Epi_Covid",
    "Perfil_centro_salud", "Perfil_residencia", "Perfil_colegio",
    "Perfil_zona_turistica", "Perfil_zona_rural",
    "Mes_Num", "Was_Promo", "Is_Future_Promo",
    "ATC_Encoded", "Temp_Media", "Temp_Desviacion", "Stock_Ratio",
]


def entrenar_modelo_ml(df_features):
    if not ML_AVAILABLE:
        return None, {"error": "XGBoost/sklearn no instalado"}, {}
    df = df_features.dropna(subset=[COL_VENTAS]).copy()
    cols_disponibles = [c for c in FEATURE_COLS if c in df.columns]
    if len(cols_disponibles) < 3 or len(df) < 20:
        return None, {"error": "Datos insuficientes para entrenar"}, {}
    X = df[cols_disponibles].fillna(0)
    y = df[COL_VENTAS].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    df_test = X_test.copy()
    df_test["y_real"] = y_test; df_test["y_pred"] = y_pred
    df_test[COL_CN] = df.loc[X_test.index, COL_CN].values
    rmse_por_cn = df_test.groupby(COL_CN).apply(
        lambda g: np.sqrt(((g["y_real"] - g["y_pred"])**2).mean())).to_dict()
    imp = dict(zip(cols_disponibles, model.feature_importances_))
    imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    metricas = {"rmse": round(rmse, 2), "r2": round(r2, 4),
                "n_train": len(X_train), "n_test": len(X_test),
                "features": cols_disponibles, "importance": imp_sorted,
                "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "n_registros_historico": len(df_features)}
    return model, metricas, rmse_por_cn


def predecir_demanda_ml(model, df_features_futuro, rmse_por_cn, nivel_servicio_pct=95):
    cols_disponibles = [c for c in FEATURE_COLS if c in df_features_futuro.columns]
    X = df_features_futuro[cols_disponibles].fillna(0)
    predicciones = model.predict(X).clip(min=0)
    z = Z_SCORES.get(nivel_servicio_pct, 1.645)
    rmse_global = np.mean(list(rmse_por_cn.values())) if rmse_por_cn else 1.0
    result = df_features_futuro[[COL_CN]].copy()
    result["Prediccion_Media"] = predicciones
    result["RMSE_Producto"] = result[COL_CN].map(rmse_por_cn).fillna(rmse_global)
    result["Safety_Stock_ML"] = np.ceil(z * result["RMSE_Producto"])
    result["Prediccion_Final"] = np.ceil(result["Prediccion_Media"] + result["Safety_Stock_ML"])
    result = result.groupby(COL_CN).agg({
        "Prediccion_Media": "mean", "RMSE_Producto": "first",
        "Safety_Stock_ML": "first", "Prediccion_Final": "mean",
    }).reset_index()
    result["Prediccion_Final"] = np.ceil(result["Prediccion_Final"])
    return result


def guardar_modelo_farmacia(model, metricas, rmse_por_cn):
    ruta = ruta_farmacia_activa()
    if ruta is None or model is None:
        return
    joblib.dump(model, ruta / "modelo_ml.joblib")
    guardar_json_farmacia("modelo_metricas.json", metricas)
    guardar_json_farmacia("modelo_rmse_cn.json", rmse_por_cn)

def cargar_modelo_farmacia():
    ruta = ruta_farmacia_activa()
    if ruta is None:
        return None, None, None
    model_path = ruta / "modelo_ml.joblib"
    if not model_path.exists():
        return None, None, None
    try:
        model = joblib.load(model_path)
        metricas = cargar_json_farmacia("modelo_metricas.json", default={})
        rmse_por_cn = cargar_json_farmacia("modelo_rmse_cn.json", default={})
        return model, metricas, rmse_por_cn
    except Exception:
        return None, None, None


# ===========================================================================
# ANALISIS, MATCHING, TIERS, PEDIDOS
# ===========================================================================
def calcular_health_score(df_inventario, df_ventas_media, meses_cobertura_objetivo=None):
    if meses_cobertura_objetivo is None:
        meses_cobertura_objetivo = HEALTH_SCORE_MESES_DEFAULT
    df = df_inventario.merge(df_ventas_media, on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    df["Stock_Ideal"] = df["Venta_Media_Mensual"] * meses_cobertura_objetivo
    mask = df["Venta_Media_Mensual"] > 0
    if mask.sum() == 0: return 50.0
    df_a = df[mask].copy()
    df_a["Desv"] = ((df_a[COL_STOCK] - df_a["Stock_Ideal"]).abs() / df_a["Stock_Ideal"]).clip(upper=2.0)
    return round(max(0, 100 - df_a["Desv"].mean() * 100), 1)

def calcular_stock_zombie(df_inventario, df_ventas, meses_sin_venta=12):
    df_inv = df_inventario.copy()
    if COL_FECHA in df_ventas.columns:
        df_v = df_ventas.copy()
        df_v[COL_FECHA] = pd.to_datetime(df_v[COL_FECHA], errors="coerce", dayfirst=True)
        fecha_corte = df_v[COL_FECHA].max() - relativedelta(months=meses_sin_venta)
        ventas_rec = df_v[df_v[COL_FECHA] >= fecha_corte].groupby(COL_CN)[COL_VENTAS].sum().reset_index()
        cn_con_ventas = set(ventas_rec[ventas_rec[COL_VENTAS] > 0][COL_CN])
    else:
        vt = df_ventas.groupby(COL_CN)[COL_VENTAS].sum().reset_index()
        cn_con_ventas = set(vt[vt[COL_VENTAS] > 0][COL_CN])
    df_z = df_inv[(df_inv[COL_STOCK] > 0) & (~df_inv[COL_CN].isin(cn_con_ventas))].copy()
    df_z["Valor_Inmovilizado"] = (df_z[COL_STOCK] * df_z.get(COL_PVL, pd.Series(0))).fillna(0)
    return df_z

def calcular_roturas(df_inventario, df_ventas_media):
    df_m = df_inventario.merge(df_ventas_media, on=COL_CN, how="left")
    df_m["Venta_Media_Mensual"] = df_m["Venta_Media_Mensual"].fillna(0)
    return df_m[(df_m[COL_STOCK] == 0) & (df_m["Venta_Media_Mensual"] > 0)]

def calcular_pareto_laboratorios(df_inventario):
    df = df_inventario.copy()
    if COL_PVL not in df.columns or COL_LAB not in df.columns: return pd.DataFrame()
    df["Valor_Stock"] = (df[COL_STOCK] * df[COL_PVL]).fillna(0)
    pareto = df.groupby(COL_LAB)["Valor_Stock"].sum().sort_values(ascending=False).reset_index()
    total = pareto["Valor_Stock"].sum()
    pareto["Pct_Acumulado"] = (pareto["Valor_Stock"].cumsum() / total * 100) if total > 0 else pd.Series(0.0, index=pareto.index)
    return pareto


# --- UVI: productos sin ventas en 6 meses ---
def calcular_stock_uvi(df_inventario, df_ventas, meses_sin_venta=6):
    """Productos con stock > 0 sin ventas en los ultimos 6 meses (pre-zombie)."""
    df_inv = df_inventario.copy()
    if COL_FECHA in df_ventas.columns:
        df_v = df_ventas.copy()
        df_v[COL_FECHA] = pd.to_datetime(df_v[COL_FECHA], errors="coerce", dayfirst=True)
        fecha_corte_uvi = df_v[COL_FECHA].max() - relativedelta(months=meses_sin_venta)
        fecha_corte_zombie = df_v[COL_FECHA].max() - relativedelta(months=12)
        ventas_rec = df_v[df_v[COL_FECHA] >= fecha_corte_uvi].groupby(COL_CN)[COL_VENTAS].sum().reset_index()
        ventas_12m = df_v[df_v[COL_FECHA] >= fecha_corte_zombie].groupby(COL_CN)[COL_VENTAS].sum().reset_index()
        cn_sin_venta_6m = set(df_inv[COL_CN]) - set(ventas_rec[ventas_rec[COL_VENTAS] > 0][COL_CN])
        cn_con_venta_12m = set(ventas_12m[ventas_12m[COL_VENTAS] > 0][COL_CN])
        # UVI = sin ventas 6 meses PERO con ventas en 12 meses (si no, ya es zombie)
        cn_uvi = cn_sin_venta_6m & cn_con_venta_12m
    else:
        return pd.DataFrame()
    df_u = df_inv[(df_inv[COL_STOCK] > 0) & (df_inv[COL_CN].isin(cn_uvi))].copy()
    if COL_PVL in df_u.columns:
        df_u["Valor_Inmovilizado"] = (df_u[COL_STOCK] * df_u[COL_PVL]).fillna(0)
    else:
        df_u["Valor_Inmovilizado"] = 0
    return df_u


# --- Rotacion de stock ---
def calcular_rotacion_stock(df_inventario, df_ventas_media):
    """Ratio de venta sobre stock. Alto = buen movimiento, bajo = sobrestock.
    Excluye productos con stock=0 (son roturas, no sobrestock)."""
    df = df_inventario.merge(df_ventas_media[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    # Solo productos CON stock (stock=0 es rotura, no va aqui)
    df = df[df[COL_STOCK] > 0].copy()
    if df.empty:
        return pd.DataFrame()
    df["Rotacion"] = (df["Venta_Media_Mensual"] / df[COL_STOCK]).round(2)
    df["Meses_Stock"] = np.where(df["Venta_Media_Mensual"] > 0,
        (df[COL_STOCK] / df["Venta_Media_Mensual"]).round(1), np.nan)
    cols = [c for c in [COL_CN, COL_NOMBRE, COL_LAB, COL_STOCK, "Venta_Media_Mensual",
            "Rotacion", "Meses_Stock"] if c in df.columns]
    return df[cols].sort_values("Rotacion", ascending=True)


# --- ROI por laboratorio ---
def calcular_roi_laboratorios(df_inventario, df_ventas_media):
    """ROI = Venta_Anual_€ / Stock_€ por laboratorio."""
    if COL_LAB not in df_inventario.columns or COL_PVL not in df_inventario.columns:
        return pd.DataFrame()
    df = df_inventario.merge(df_ventas_media[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    df["Stock_EUR"] = (df[COL_STOCK] * df[COL_PVL]).fillna(0)
    df["Venta_Anual_EUR"] = (df["Venta_Media_Mensual"] * 12 * df[COL_PVL]).fillna(0)
    lab_agg = df.groupby(COL_LAB).agg(
        Stock_EUR=("Stock_EUR", "sum"),
        Venta_Anual_EUR=("Venta_Anual_EUR", "sum"),
        N_Productos=(COL_CN, "count"),
    ).reset_index()
    lab_agg["ROI"] = np.where(lab_agg["Stock_EUR"] > 0,
        (lab_agg["Venta_Anual_EUR"] / lab_agg["Stock_EUR"]).round(2), 0.0)
    lab_agg = lab_agg.sort_values("ROI", ascending=False)
    return lab_agg


# --- Coste de oportunidad de roturas ---
def calcular_coste_oportunidad(df_inventario, df_ventas_media):
    """Euros perdidos al mes por no tener stock de productos con demanda."""
    df_rot = calcular_roturas(df_inventario, df_ventas_media)
    if df_rot.empty:
        return 0.0, pd.DataFrame()
    df_rot = df_rot.copy()
    if COL_PVL in df_rot.columns:
        df_rot["Venta_Perdida_Mes"] = (df_rot["Venta_Media_Mensual"] * df_rot[COL_PVL]).fillna(0).round(2)
    else:
        df_rot["Venta_Perdida_Mes"] = 0.0
    total = df_rot["Venta_Perdida_Mes"].sum()
    cols = [c for c in [COL_CN, COL_NOMBRE, COL_LAB, "Venta_Media_Mensual",
            COL_PVL, "Venta_Perdida_Mes"] if c in df_rot.columns]
    return round(total, 2), df_rot[cols].sort_values("Venta_Perdida_Mes", ascending=False)


# --- Benchmark HS vs red ---
def calcular_benchmark_hs(mi_hs):
    """Compara el HS de la farmacia activa con la media de la red."""
    red = cargar_red_config()
    farmacias = red.get("farmacias_activas", [])
    farmacia_activa = st.session_state.get("farmacia_activa", "")
    if len(farmacias) < 2:
        return {"mi_hs": mi_hs, "media_red": None, "n_farmacias": 0}
    scores = []
    for farm in farmacias:
        if farm == farmacia_activa:
            continue
        path = BASE_DIR / farm / "historico_kpi.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    hist = json.load(f)
                if hist:
                    scores.append(hist[-1].get("health_score", 0))
            except Exception:
                pass
    if not scores:
        return {"mi_hs": mi_hs, "media_red": None, "n_farmacias": 0}
    return {
        "mi_hs": mi_hs,
        "media_red": round(np.mean(scores), 1),
        "n_farmacias": len(scores),
    }

# --- Nuevas Métricas Auditoría ---
def calcular_indice_servicio(df_inventario, df_ventas):
    """Calcula el Fill Rate: % de productos con demanda habitual que NO están en rotura."""
    if df_ventas.empty or COL_FECHA not in df_ventas.columns: return 100.0, 0, 0
    df_v = df_ventas.copy()
    df_v[COL_FECHA] = pd.to_datetime(df_v[COL_FECHA], errors="coerce", dayfirst=True)
    
    fecha_reciente = df_v[COL_FECHA].max()
    if pd.isna(fecha_reciente): return 100.0, 0, 0
    
    fecha_corte = fecha_reciente - relativedelta(months=3)
    ventas_recientes = df_v[df_v[COL_FECHA] >= fecha_corte].groupby(COL_CN)[COL_VENTAS].sum().reset_index()
    cn_demanda = set(ventas_recientes[ventas_recientes[COL_VENTAS] > 0][COL_CN])
    if not cn_demanda: return 100.0, 0, 0
    
    df_i = df_inventario[df_inventario[COL_CN].isin(cn_demanda)]
    cn_con_stock = set(df_i[df_i[COL_STOCK] > 0][COL_CN])
    
    fill_rate = len(cn_con_stock) / len(cn_demanda) * 100
    return fill_rate, len(cn_demanda), len(cn_demanda) - len(cn_con_stock)

def calcular_analisis_abc(df_inventario, df_ventas_media):
    """Clasifica los productos en A (80% valor stock), B (15%), C (5%)."""
    if COL_PVL not in df_inventario.columns or df_inventario.empty: return pd.DataFrame()
    df = df_inventario.merge(df_ventas_media[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    df["Valor_Stock"] = (df[COL_STOCK] * df[COL_PVL]).fillna(0)
    
    df = df.sort_values(by="Valor_Stock", ascending=False).reset_index(drop=True)
    total_valor = df["Valor_Stock"].sum()
    if total_valor == 0: return df
    
    df["Pct_Valor_Acumulado"] = (df["Valor_Stock"].cumsum() / total_valor) * 100
    df["Clasificacion_ABC"] = np.where(df["Pct_Valor_Acumulado"] <= 80, "A (80%)",
                                np.where(df["Pct_Valor_Acumulado"] <= 95, "B (15%)", "C (5%)"))
    return df

def calcular_riesgo_caducidad(df_inventario):
    """Simula o calcula el valor en riesgo por caducidad en < 6 meses."""
    df = df_inventario.copy()
    if df.empty: return 0.0, 0, False
    
    if "Fecha_Caducidad" in df.columns:
        df["Fecha_Caducidad"] = pd.to_datetime(df["Fecha_Caducidad"], errors="coerce")
        limite = datetime.now() + relativedelta(months=6)
        df_cad = df[df["Fecha_Caducidad"] <= limite]
        valor = (df_cad[COL_STOCK] * df_cad.get(COL_PVL, pd.Series(0))).fillna(0).sum()
        return valor, len(df_cad), True
    else:
        # MOCK realista para auditorías donde aún no se sube la caducidad (se asume un 3% del stock inmovilizado zombie/UVI)
        if COL_PVL in df.columns:
            valor_total = (df[COL_STOCK] * df[COL_PVL]).fillna(0).sum()
            riesgo_estimado = valor_total * 0.03  # 3% historico de caducos perdidos
            prods = max(1, len(df) // 40)
            return riesgo_estimado, prods, False
        return 0.0, 0, False

def calcular_matriz_rentabilidad_gmroi(df_inventario, df_ventas_media):
    """Calcula GMROI proxy (Margen * Rotación) si existe PVP y PVL."""
    if COL_PVL not in df_inventario.columns or df_inventario.empty: return pd.DataFrame(), False
        
    df = df_inventario.copy()
    df = df.merge(df_ventas_media[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    
    df["PVP_Calc"] = pd.to_numeric(df.get("PVP", pd.Series(np.nan)), errors="coerce")
    df["PVL_Calc"] = pd.to_numeric(df.get(COL_PVL, pd.Series(np.nan)), errors="coerce")
    tiene_pvp_real = "PVP" in df.columns and df["PVP_Calc"].notna().sum() > 0
    df["PVP_Calc"] = df["PVP_Calc"].fillna(df["PVL_Calc"] * 1.3).fillna(0)
    
    # Margen Bruto estimado
    df["Margen_Bruto_Pct"] = np.where(df["PVP_Calc"] > 0, (df["PVP_Calc"] - df["PVL_Calc"]) / df["PVP_Calc"], 0)
    
    # Rotacion proxy
    df["Rotacion_Proxy"] = np.where(df[COL_STOCK] > 0, df["Venta_Media_Mensual"] / df[COL_STOCK], df["Venta_Media_Mensual"])
    df["GMROI"] = df["Margen_Bruto_Pct"] * df["Rotacion_Proxy"]
    
    margen_mediano = df[df["Margen_Bruto_Pct"]>0]["Margen_Bruto_Pct"].median() if not df[df["Margen_Bruto_Pct"]>0].empty else 0.3
    rot_mediana = df[df["Rotacion_Proxy"]>0]["Rotacion_Proxy"].median() if not df[df["Rotacion_Proxy"]>0].empty else 1.0
    
    cond_alta_rot = df["Rotacion_Proxy"] >= rot_mediana
    cond_alto_mar = df["Margen_Bruto_Pct"] >= margen_mediano
    
    df["Cuadrante"] = np.where(cond_alta_rot & cond_alto_mar, "Estrellas (Alto M. / Alta R.)",
                      np.where(cond_alta_rot & ~cond_alto_mar, "Genera Tráfico (Bajo M. / Alta R.)",
                      np.where(~cond_alta_rot & cond_alto_mar, "Rentables Lentos (Alto M. / Baja R.)",
                               "Problema (Bajo M. / Baja R.)")))
    return df, tiene_pvp_real

def calcular_dependencia_estacional(df_ventas):
    """Calcula el % de ventas que concentra cada mes para detectar picos brutales (estacionalidad)."""
    if df_ventas.empty or COL_FECHA not in df_ventas.columns: return pd.DataFrame()
    df_v = df_ventas.copy()
    df_v[COL_FECHA] = pd.to_datetime(df_v[COL_FECHA], errors="coerce", dayfirst=True)
    df_v["Mes"] = df_v[COL_FECHA].dt.month
    df_v["Anio"] = df_v[COL_FECHA].dt.year
    
    ventas_por_mes_anio = df_v.groupby(["Anio", "Mes"])[COL_VENTAS].sum().reset_index()
    promedio_mensual = ventas_por_mes_anio.groupby("Mes")[COL_VENTAS].mean().reset_index()
    total_ventas_promedio_anual = promedio_mensual[COL_VENTAS].sum()
    
    if total_ventas_promedio_anual == 0: return pd.DataFrame()
    
    promedio_mensual["Pct_Ventas"] = (promedio_mensual[COL_VENTAS] / total_ventas_promedio_anual) * 100
    promedio_mensual["Mes_Nombre"] = promedio_mensual["Mes"].map({
        1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun",
        7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic"})
    return promedio_mensual.sort_values("Mes")

def calcular_conciliacion_fisico_logico(df_teorico, df_fisico):
    """Cruza Stock_Teorico con Cantidad_Fisica del CSV con la pistola."""
    col_cn = [c for c in df_fisico.columns if "cod" in c.lower() or "cn" in c.lower()]
    col_cant = [c for c in df_fisico.columns if "cant" in c.lower() or "stock" in c.lower()]
    if not col_cn or not col_cant: return None
    
    c_cn, c_qt = col_cn[0], col_cant[0]
    agg_fisico = df_fisico.groupby(c_cn)[c_qt].sum().reset_index()
    agg_fisico.columns = [COL_CN, "Stock_Fisico"]
    
    df = df_teorico.copy()
    if COL_PVL not in df.columns: df[COL_PVL] = 0
    df = df.merge(agg_fisico, on=COL_CN, how="outer")
    df[COL_STOCK] = df[COL_STOCK].fillna(0)
    df["Stock_Fisico"] = df["Stock_Fisico"].fillna(0)
    
    df["Descuadre_Uds"] = df["Stock_Fisico"] - df[COL_STOCK]
    df["Descuadre_Eur"] = df["Descuadre_Uds"] * df[COL_PVL]
    return df

def calcular_flujo_caja_inventario(df_inv, df_vm, dias_pago_media, df_facturas=None):
    """Cruza la rotacion del inventario con los dias de pago para ver el Flujo de Caja."""
    if df_facturas is not None and not df_facturas.empty:
        c_imp = [c for c in df_facturas.columns if "import" in c.lower() or "total" in c.lower() or "eur" in c.lower()]
        c_dias = [c for c in df_facturas.columns if "dias" in c.lower() or "pago" in c.lower()]
        if c_imp and c_dias:
            df_facturas["Importe_Calc"] = pd.to_numeric(df_facturas[c_imp[0]], errors="coerce").fillna(0)
            df_facturas["Dias_Calc"] = pd.to_numeric(df_facturas[c_dias[0]], errors="coerce").fillna(0)
            sum_imp = df_facturas["Importe_Calc"].sum()
            if sum_imp > 0:
                df_facturas["Peso"] = df_facturas["Importe_Calc"] * df_facturas["Dias_Calc"]
                dias_pago_media = df_facturas["Peso"].sum() / sum_imp
            
    if COL_PVL not in df_inv.columns or df_inv.empty: return dias_pago_media, 0, 0, 0, 0
    
    df = df_inv.merge(df_vm[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    
    valor_stock_total = (df[COL_STOCK] * df[COL_PVL]).fillna(0).sum()
    coste_ventas_mensual = (df["Venta_Media_Mensual"] * df[COL_PVL]).fillna(0).sum()
    
    if coste_ventas_mensual == 0: return dias_pago_media, 0, 0, 0, 0
    
    coste_ventas_diario = coste_ventas_mensual / 30
    dias_inventario = valor_stock_total / coste_ventas_diario
    
    ccc = dias_inventario - dias_pago_media
    necesidad_fm = ccc * coste_ventas_diario if ccc > 0 else 0
    flotador = abs(ccc) * coste_ventas_diario if ccc < 0 else 0
    
    return dias_pago_media, dias_inventario, ccc, necesidad_fm, flotador


# --- Panel de surtido ---
def generar_panel_surtido(df_inventario, reglas=None):
    """Por cada molecula o regla granulada, cuantos labs distintos tienen stock."""
    if COL_NOMBRE not in df_inventario.columns or COL_LAB not in df_inventario.columns:
        return pd.DataFrame()
    df = df_inventario[df_inventario[COL_STOCK] > 0].copy()
    if df.empty:
        return pd.DataFrame()
        
    df["Tipo_Prod"] = df.apply(lambda r: heuristica_tipo_producto(r.get(COL_CN, ""), r.get(COL_MOLECULA if COL_MOLECULA in df.columns else COL_NOMBRE, "")), axis=1)
    
    # Extraer lista de reglas en texto
    reglas_str = []
    reglas_map = {}
    if reglas:
        for r in reglas:
            r_str = str(r.get("presentacion", "")).lower().strip()
            if r_str:
                reglas_str.append(r_str)
                reglas_map[r_str] = int(float(r.get("min_labs", 1)))
                
    # Asignar grupo dinamico
    col_mol = COL_MOLECULA if COL_MOLECULA in df.columns else COL_NOMBRE
    df["Surtido_Grupo"] = df.apply(lambda r: asignar_grupo_surtido(r[COL_NOMBRE], r[col_mol], reglas_str), axis=1)

    panel = df.groupby("Surtido_Grupo", as_index=False).agg(
        N_Labs=(COL_LAB, "nunique"),
        Labs=(COL_LAB, lambda x: ", ".join(sorted([str(i) for i in x.unique() if pd.notna(i)]))),
        Stock_Total=(COL_STOCK, "sum"),
        Tipo=( "Tipo_Prod", lambda x: ", ".join(sorted(list(set(x)))))
    ).rename(columns={"Surtido_Grupo": "Presentacion/Molecula"}).sort_values("N_Labs", ascending=True)

    # Aplicar reglas si existen
    if reglas:
        panel["Regla_Min"] = panel["Presentacion/Molecula"].str.lower().map(reglas_map).fillna(0).astype(int)
        panel["Cumple"] = panel.apply(
            lambda r: "✅" if r["Regla_Min"] == 0 or r["N_Labs"] >= r["Regla_Min"] else "❌", axis=1)
    else:
        panel["Regla_Min"] = 0
        panel["Cumple"] = "—"
    return panel


# --- Validacion post-pedido de surtido ---
def validar_surtido_pedido(df_inventario, df_pedido, reglas):
    """Comprueba si el pedido mantiene la diversificacion minima preasignando grupos."""
    alertas = []
    if not reglas or df_pedido is None or df_pedido.empty:
        return alertas
    if COL_NOMBRE not in df_inventario.columns or COL_LAB not in df_inventario.columns:
        return alertas
    
    # Estado actual de surtido + Pedidos (Ficticio/Simulado juntos)
    df_stock = df_inventario[df_inventario[COL_STOCK] > 0].copy()
    
    reglas_str = []
    for r in reglas:
        r_str = str(r.get("presentacion", "")).lower().strip()
        if r_str: reglas_str.append(r_str)
        
    col_mol = COL_MOLECULA if COL_MOLECULA in df_stock.columns else COL_NOMBRE
    df_stock["Surtido_Grupo"] = df_stock.apply(lambda r: asignar_grupo_surtido(r[COL_NOMBRE], r[col_mol], reglas_str), axis=1)
    
    surtido_actual = df_stock.groupby("Surtido_Grupo")[COL_LAB].nunique().to_dict()
    
    # Verificar reglas
    for regla in reglas:
        mol = str(regla.get("presentacion", "")).strip().lower()
        min_labs = int(float(regla.get("min_labs", 1)))
        if not mol or min_labs <= 0:
            continue
        labs_actuales = surtido_actual.get(mol, 0)
        if labs_actuales < min_labs:
            alertas.append({
                "molecula": mol.title(),
                "min_labs": min_labs,
                "labs_actuales": labs_actuales,
                "mensaje": f"⚠️ {mol.title()}: tienes {labs_actuales} lab(s), regla exige mín. {min_labs}",
            })
    return alertas


# --- Informe HTML descargable ---
def generar_informe_pdf(farmacia_nombre, hs, n_zombies, valor_zombie, n_uvi, valor_uvi,
                       n_roturas, coste_oportunidad, ahorro_acum, rotacion_media, benchmark):
    """Genera un informe PDF descargable para el titular."""
    fecha = datetime.now().strftime("%d/%m/%Y")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Titulo
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 102, 255)
    pdf.cell(0, 12, "PharmaFlow - Informe de Estado", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(0, 196, 154)
    pdf.set_line_width(1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    # Farmacia y fecha
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, f"Farmacia: {farmacia_nombre}  |  Fecha: {fecha}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    # Benchmark
    if benchmark and benchmark.get("media_red") is not None:
        diff = hs - benchmark["media_red"]
        signo = "+" if diff >= 0 else ""
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 7, f"Benchmark Red: Tu {hs}% vs Media {benchmark['media_red']}% ({benchmark['n_farmacias']} farmacias) ({signo}{diff:.1f}%)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    # KPIs tabla
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "Indicadores Clave", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    kpis = [
        ("Health Score", f"{hs}%"),
        ("Rotacion Media", f"{rotacion_media:.2f}"),
        ("Ahorro Acumulado", f"{ahorro_acum:,.2f} EUR"),
        ("Stock Zombie", f"{valor_zombie:,.2f} EUR ({n_zombies} prods)"),
        ("Stock UVI", f"{valor_uvi:,.2f} EUR ({n_uvi} prods)"),
        ("Roturas", f"{n_roturas} prods ({coste_oportunidad:,.2f} EUR/mes)"),
    ]
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(248, 249, 252)
    col_w = 90
    for i, (label, value) in enumerate(kpis):
        fill = i % 2 == 0
        pdf.cell(col_w, 8, f"  {label}", border=0, fill=fill)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(col_w, 8, value, border=0, fill=fill, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 10)
    pdf.ln(6)
    # Resumen
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Resumen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    resumen = [
        f"Capital inmovilizado en productos sin movimiento: {valor_zombie + valor_uvi:,.2f} EUR",
        f"Coste de oportunidad por roturas: {coste_oportunidad:,.2f} EUR/mes",
        f"Ahorro generado con PharmaFlow: {ahorro_acum:,.2f} EUR",
    ]
    for linea in resumen:
        pdf.cell(5, 7, "-")
        pdf.cell(0, 7, linea, new_x="LMARGIN", new_y="NEXT")
    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Generado por PharmaFlow v{VERSION} | {fecha}", align="C")
    return bytes(pdf.output())


# --- Matching indexado ---
def construir_indice_ofertas(df_ofertas):
    indice = {}
    if df_ofertas is None or df_ofertas.empty or "Oferta_Nombre" not in df_ofertas.columns: return indice
    for idx, row in df_ofertas.iterrows():
        nombre = str(row.get("Oferta_Nombre", "")).lower().strip()
        if len(nombre) < 3: continue
        for token in nombre.split():
            if len(token) >= 3:
                indice.setdefault(token, []).append(idx)
    return indice

def _token_contenido(token_oferta, tokens_disponibles):
    return any(token_oferta in tp for tp in tokens_disponibles)

def buscar_oferta_por_indice(nombre_producto, molecula_producto, indice, df_ofertas):
    tokens_prod = set(nombre_producto.lower().strip().split())
    tokens_mol = set(molecula_producto.lower().strip().split())
    tokens_disponibles = tokens_prod | tokens_mol
    candidatas_idx = set()
    for token in tokens_disponibles:
        if token in indice: candidatas_idx.update(indice[token])
        for idx_token, idx_list in indice.items():
            if idx_token in token or token in idx_token:
                candidatas_idx.update(idx_list)
    for oidx in candidatas_idx:
        oferta = df_ofertas.loc[oidx]
        nombre_oferta = str(oferta.get("Oferta_Nombre", "")).lower().strip()
        if len(nombre_oferta) < 3: continue
        tokens_oferta = set(nombre_oferta.split())
        if all(_token_contenido(to, tokens_disponibles) for to in tokens_oferta):
            return oferta
    return None

# --- Tiers dinamicos ---
def aplicar_tiers_dinamicos(cantidad, tiers):
    if tiers is None or (hasattr(tiers, '__len__') and len(tiers) == 0): return cantidad, 0.0, "Sin oferta", False
    for i in range(len(tiers)-1, -1, -1):
        tier = tiers[i]; tier_num = i + 1
        if cantidad >= tier["min"]: return cantidad, tier["dto"], f"Tier {tier_num}", False
        if cantidad >= tier["min"] * 0.8: return tier["min"], tier["dto"], f"Tier {tier_num} (Upselling)", True
    return cantidad, 0.0, "Sin oferta", False

def aplicar_ofertas_indexado(df_pedido, df_ofertas=None):
    df = df_pedido.copy()
    df["Descuento_Aplicado"] = 0.0; df["Tier_Aplicado"] = "Sin oferta"; df["Upselling"] = False
    if df_ofertas is None or df_ofertas.empty or "Oferta_Nombre" not in df_ofertas.columns: return df
    indice = construir_indice_ofertas(df_ofertas)
    for idx, row in df.iterrows():
        oferta = buscar_oferta_por_indice(str(row.get(COL_NOMBRE, "")), str(row.get(COL_MOLECULA, "")), indice, df_ofertas)
        if oferta is None: continue
        tiers = oferta.get("Tiers", [])
        if isinstance(tiers, str):
            try: tiers = json.loads(tiers)
            except (json.JSONDecodeError, ValueError, TypeError): tiers = []
        cant_final, dto, tier_nombre, es_ups = aplicar_tiers_dinamicos(int(row["Cantidad_A_Pedir"]), tiers)
        df.at[idx, "Cantidad_A_Pedir"] = cant_final
        df.at[idx, "Descuento_Aplicado"] = dto
        df.at[idx, "Tier_Aplicado"] = tier_nombre
        df.at[idx, "Upselling"] = es_ups
    return df

# --- Generadores de pedido ---
def generar_pedido_cobertura(df_inventario, df_ventas_media, meses_cobertura,
                             df_ofertas=None, nivel_servicio=1.65, productos_protegidos=None,
                             restriccion_caducidad_dias=None):
    df = df_inventario.merge(df_ventas_media, on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    df["Venta_Std_Mensual"] = df.get("Venta_Std_Mensual", pd.Series(0, index=df.index)).fillna(0)
    df["Safety_Stock"] = np.ceil(nivel_servicio * df["Venta_Std_Mensual"] * np.sqrt(meses_cobertura))
    df["Stock_Objetivo"] = np.ceil(df["Venta_Media_Mensual"] * meses_cobertura + df["Safety_Stock"])

    if restriccion_caducidad_dias is not None and "Caducidad_Dias" in df.columns:
        meses_hasta_cad = (df["Caducidad_Dias"] / 30).clip(lower=0)
        tope_cad = np.ceil(df["Venta_Media_Mensual"] * meses_hasta_cad)
        mask_cad = meses_hasta_cad < meses_cobertura
        df.loc[mask_cad, "Stock_Objetivo"] = np.minimum(
            df.loc[mask_cad, "Stock_Objetivo"], tope_cad[mask_cad])

    df["Cantidad_A_Pedir"] = np.maximum(0, df["Stock_Objetivo"] - df[COL_STOCK]).astype(int)

    if productos_protegidos:
        for pp in productos_protegidos:
            cn = pp.get("codigo_nacional", ""); stock_min = int(float(pp.get("stock_minimo", 1)))
            mask = df[COL_CN] == cn
            if mask.any():
                if df.loc[mask, COL_STOCK].iloc[0] < stock_min:
                    df.loc[mask, "Cantidad_A_Pedir"] = df.loc[mask, "Cantidad_A_Pedir"].clip(lower=stock_min - df.loc[mask, COL_STOCK].iloc[0])
            else:
                nueva = {COL_CN: cn, COL_STOCK: 0, "Cantidad_A_Pedir": stock_min,
                         "Venta_Media_Mensual": 0, "Safety_Stock": 0, "Stock_Objetivo": stock_min,
                         COL_NOMBRE: pp.get("nombre", cn)}
                df = pd.concat([df, pd.DataFrame([nueva])], ignore_index=True)

    df_ped = df[df["Cantidad_A_Pedir"] > 0].copy()
    df_ped["Precio_Unitario"] = df_ped[COL_PVL].fillna(0) if COL_PVL in df_ped.columns else 0.0
    df_ped = aplicar_ofertas_indexado(df_ped, df_ofertas)
    df_ped["Coste_Sin_Dto"] = df_ped["Cantidad_A_Pedir"] * df_ped["Precio_Unitario"]
    df_ped["Coste_Con_Dto"] = df_ped["Coste_Sin_Dto"] * (1 - df_ped["Descuento_Aplicado"])
    df_ped["Ahorro"] = df_ped["Coste_Sin_Dto"] - df_ped["Coste_Con_Dto"]
    return df_ped

def generar_pedido_presupuesto(df_inventario, df_ventas_media, presupuesto, meses_cobertura,
                                df_ofertas=None, productos_protegidos=None):
    df_ideal = generar_pedido_cobertura(df_inventario, df_ventas_media, meses_cobertura, df_ofertas,
                                         productos_protegidos=productos_protegidos)
    if df_ideal.empty: return df_ideal
    cn_prot = {pp["codigo_nacional"] for pp in (productos_protegidos or [])}
    df_p = df_ideal[df_ideal[COL_CN].isin(cn_prot)].copy()
    df_r = df_ideal[~df_ideal[COL_CN].isin(cn_prot)].copy()
    gasto_p = (df_p["Cantidad_A_Pedir"] * df_p["Precio_Unitario"] * (1 - df_p["Descuento_Aplicado"])).sum() if not df_p.empty else 0
    ppto_r = max(0, presupuesto - gasto_p)
    df_r = df_r.sort_values("Venta_Media_Mensual", ascending=False).reset_index(drop=True)
    gasto = 0.0; cantidades = []
    for _, row in df_r.iterrows():
        precio_neto = row["Precio_Unitario"] * (1 - row["Descuento_Aplicado"])
        if precio_neto <= 0: cantidades.append(int(row["Cantidad_A_Pedir"])); continue
        margen = ppto_r - gasto
        if margen <= 0: cantidades.append(0); continue
        uds = min(int(row["Cantidad_A_Pedir"]), int(margen / precio_neto))
        cantidades.append(uds); gasto += uds * precio_neto
    df_r["Cantidad_A_Pedir"] = cantidades
    df_r = df_r[df_r["Cantidad_A_Pedir"] > 0]
    df_f = pd.concat([df_p, df_r], ignore_index=True)
    df_f["Coste_Sin_Dto"] = df_f["Cantidad_A_Pedir"] * df_f["Precio_Unitario"]
    df_f["Coste_Con_Dto"] = df_f["Coste_Sin_Dto"] * (1 - df_f["Descuento_Aplicado"])
    df_f["Ahorro"] = df_f["Coste_Sin_Dto"] - df_f["Coste_Con_Dto"]
    return df_f

def generar_pedido_ml(df_inventario, model, df_features_futuro, rmse_por_cn,
                      meses_cobertura, nivel_servicio_pct, df_ofertas=None,
                      productos_protegidos=None):
    pred = predecir_demanda_ml(model, df_features_futuro, rmse_por_cn, nivel_servicio_pct)
    df_vm_ml = pred[[COL_CN, "Prediccion_Final", "RMSE_Producto"]].copy()
    df_vm_ml = df_vm_ml.rename(columns={
        "Prediccion_Final": "Venta_Media_Mensual",
        "RMSE_Producto": "Venta_Std_Mensual",
    })
    return generar_pedido_cobertura(
        df_inventario, df_vm_ml, meses_cobertura, df_ofertas,
        nivel_servicio=0.0,
        productos_protegidos=productos_protegidos)


def predecir_demanda_ensemble(model, df_features_futuro, rmse_por_cn,
                               df_ventas_media_heuristico, nivel_servicio_pct=95):
    """Motor Ensemble: pondera ML + Heuristico por confianza por producto.
    Confianza ML = 1 - (RMSE_producto / std_ventas_producto). Si RMSE < std, ML es mejor.
    """
    pred_ml = predecir_demanda_ml(model, df_features_futuro, rmse_por_cn, nivel_servicio_pct)

    if df_ventas_media_heuristico is None or df_ventas_media_heuristico.empty:
        return pred_ml

    # Merge con heuristico
    heur = df_ventas_media_heuristico[[COL_CN, "Venta_Media_Mensual"]].copy()
    heur = heur.rename(columns={"Venta_Media_Mensual": "Pred_Heuristico"})
    result = pred_ml.merge(heur, on=COL_CN, how="left")
    result["Pred_Heuristico"] = result["Pred_Heuristico"].fillna(result["Prediccion_Media"])

    # Calcular peso ML por producto basado en confianza
    # Si std del producto es conocida, usar Std. Sino, usar RMSE como proxy.
    if "Venta_Std_Mensual" in df_ventas_media_heuristico.columns:
        std_map = df_ventas_media_heuristico.set_index(COL_CN)["Venta_Std_Mensual"].to_dict()
        result["_std"] = result[COL_CN].map(std_map).fillna(result["RMSE_Producto"])
    else:
        result["_std"] = result["RMSE_Producto"]

    # Peso ML = clip(1 - RMSE/std, 0.2, 0.8) — siempre damos algo de peso a ambos
    std_clipped = result["_std"].clip(lower=0.1)
    result["_peso_ml"] = (1 - result["RMSE_Producto"] / std_clipped).clip(0.2, 0.8)

    # Prediccion ensemble
    result["Prediccion_Media"] = (
        result["_peso_ml"] * result["Prediccion_Media"] +
        (1 - result["_peso_ml"]) * result["Pred_Heuristico"]
    )
    result["Prediccion_Final"] = np.ceil(
        result["_peso_ml"] * result["Prediccion_Final"] +
        (1 - result["_peso_ml"]) * result["Pred_Heuristico"]
    )

    result = result.drop(columns=["_std", "_peso_ml", "Pred_Heuristico"], errors="ignore")
    return result


def generar_pedido_ensemble(df_inventario, model, df_features_futuro, rmse_por_cn,
                            df_ventas_media_heuristico, meses_cobertura,
                            nivel_servicio_pct, df_ofertas=None, productos_protegidos=None):
    """Genera pedido usando motor Ensemble (ML + Heuristico ponderado)."""
    pred = predecir_demanda_ensemble(
        model, df_features_futuro, rmse_por_cn,
        df_ventas_media_heuristico, nivel_servicio_pct)
    df_vm = pred[[COL_CN, "Prediccion_Final", "RMSE_Producto"]].copy()
    df_vm = df_vm.rename(columns={
        "Prediccion_Final": "Venta_Media_Mensual",
        "RMSE_Producto": "Venta_Std_Mensual",
    })
    return generar_pedido_cobertura(
        df_inventario, df_vm, meses_cobertura, df_ofertas,
        nivel_servicio=0.0,
        productos_protegidos=productos_protegidos)


# ===========================================================================
# EXPORTACION Y GRAFICOS
# ===========================================================================
def exportar_pedido_excel(df_pedido):
    output = BytesIO()
    cols_export = [COL_CN, COL_NOMBRE, COL_LAB, COL_STOCK, "Venta_Media_Mensual", "Safety_Stock",
        "Cantidad_A_Pedir", "Precio_Unitario", "Descuento_Aplicado", "Coste_Con_Dto", "Ahorro", "Tier_Aplicado"]
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        col_lab = COL_LAB if COL_LAB in df_pedido.columns else None
        if col_lab and df_pedido[col_lab].nunique() > 1:
            for lab in sorted(df_pedido[col_lab].dropna().unique()):
                dl = df_pedido[df_pedido[col_lab] == lab]
                cv = [c for c in cols_export if c in dl.columns]
                dl[cv].to_excel(writer, sheet_name=str(lab)[:31].replace("/","-"), index=False)
        else:
            cv = [c for c in cols_export if c in df_pedido.columns]
            lab_name = str(df_pedido[col_lab].iloc[0])[:31] if col_lab and not df_pedido.empty else "Pedido"
            df_pedido[cv].to_excel(writer, sheet_name=lab_name, index=False)
    output.seek(0)
    return output

def grafico_gauge_health(score):
    bar_color = COLORS["success"] if score >= 70 else (COLORS["warning"] if score >= 40 else COLORS["danger"])
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 42, "color": COLORS["text"]}},
        gauge={"axis": {"range": [0,100]}, "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#F1F5F9", "borderwidth": 0,
            "steps": [{"range":[0,40],"color":"#FEE2E2"},{"range":[40,70],"color":"#FEF3C7"},{"range":[70,100],"color":"#D1FAE5"}],
            "threshold": {"line":{"color":COLORS["text"],"width":3},"thickness":0.8,"value":score}},
        title={"text": "Stock Health Score", "font": {"size": 16, "color": COLORS["muted"]}}))
    fig.update_layout(height=280, margin=dict(l=30,r=30,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"})
    return fig

def grafico_calendario_reposicion(df_inv, df_ventas_media):
    """Timeline de fechas estimadas de reposicion por laboratorio."""
    if COL_LAB not in df_inv.columns:
        return None
    df_m = df_inv.merge(df_ventas_media[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df_m["Venta_Media_Mensual"] = df_m["Venta_Media_Mensual"].fillna(0)
    df_m["Dias_Stock"] = np.where(df_m["Venta_Media_Mensual"] > 0,
        (df_m[COL_STOCK] / (df_m["Venta_Media_Mensual"] / 30.44)), 999)
    # Minimo dias por laboratorio (cuello de botella)
    lab_dias = df_m[df_m["Dias_Stock"] < 999].groupby(COL_LAB)["Dias_Stock"].min().reset_index()
    lab_dias = lab_dias.sort_values("Dias_Stock").head(15)
    if lab_dias.empty:
        return None
    hoy = date.today()
    lab_dias["Fecha_Reposicion"] = lab_dias["Dias_Stock"].apply(
        lambda d: (hoy + timedelta(days=max(0, d - 7))).strftime("%d/%m/%Y"))
    lab_dias["Dias_Stock"] = lab_dias["Dias_Stock"].round(0).astype(int)
    # Colores por urgencia
    colores = []
    for d in lab_dias["Dias_Stock"]:
        if d <= 14: colores.append(COLORS["danger"])
        elif d <= 30: colores.append(COLORS["warning"])
        elif d <= 60: colores.append("#FFA726")
        else: colores.append(COLORS["success"])
    lab_dias_sorted = lab_dias.sort_values("Dias_Stock", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lab_dias_sorted["Dias_Stock"],
        y=lab_dias_sorted[COL_LAB],
        orientation="h",
        marker_color=colores,
        text=[f"{d}d -> {f}" for d, f in zip(lab_dias_sorted["Dias_Stock"], lab_dias_sorted["Fecha_Reposicion"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Stock restante: %{x} dias<extra></extra>",
    ))
    fig.add_vline(x=30, line_dash="dash", line_color=COLORS["warning"],
        annotation_text="30 dias", annotation_position="top right")
    fig.update_layout(
        title={"text": "Calendario de Reposicion", "font": {"size": 16}},
        height=max(280, len(lab_dias_sorted) * 30),
        margin=dict(l=50, r=80, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        xaxis=dict(title="Dias de stock restante", gridcolor="#E2E8F0"),
        yaxis=dict(title=""),
    )
    return fig

def grafico_pareto_laboratorios(df_pareto):
    if df_pareto.empty: return go.Figure().update_layout(title="Sin datos")
    df_top = df_pareto.head(15).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_top[COL_LAB], y=df_top["Valor_Stock"], marker_color=COLORS["primary"], name="Valor Stock", opacity=0.85))
    fig.add_trace(go.Scatter(x=df_top[COL_LAB], y=df_top["Pct_Acumulado"], name="% Acum", yaxis="y2", line=dict(color=COLORS["warning"],width=3)))
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["danger"], annotation_text="80%", yref="y2", opacity=0.6)
    fig.update_layout(title={"text":"Pareto — Stock por Lab","font":{"size":16}},
        yaxis=dict(title="\u20ac",gridcolor="#E2E8F0"), yaxis2=dict(title="%",overlaying="y",side="right",range=[0,105]),
        height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"},
        xaxis=dict(tickangle=-45), barmode="group", margin=dict(l=50,r=50,t=50,b=80))
    return fig

def grafico_historico_kpi(df_hist_kpi):
    if df_hist_kpi.empty: return None, None
    df_h = df_hist_kpi.copy()
    fig_hs = go.Figure()
    fig_hs.add_trace(go.Scatter(x=df_h["fecha"], y=df_h["health_score"], mode="lines+markers",
        line=dict(color=COLORS["primary"],width=3), marker=dict(size=8)))
    fig_hs.update_layout(title={"text":"Evolucion Health Score","font":{"size":16}}, yaxis=dict(range=[0,100],title="%",gridcolor="#E2E8F0"),
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"})
    # Evolucion zombies y roturas (en vez de ahorro duplicado)
    fig_zr = go.Figure()
    if "n_zombies" in df_h.columns:
        fig_zr.add_trace(go.Scatter(x=df_h["fecha"], y=df_h["n_zombies"], mode="lines+markers",
            name="Zombies", line=dict(color=COLORS["warning"], width=2), marker=dict(size=6)))
    if "n_roturas" in df_h.columns:
        fig_zr.add_trace(go.Scatter(x=df_h["fecha"], y=df_h["n_roturas"], mode="lines+markers",
            name="Roturas", line=dict(color=COLORS["danger"], width=2), marker=dict(size=6)))
    fig_zr.update_layout(title={"text":"Evolucion Zombies y Roturas","font":{"size":16}},
        yaxis=dict(title="Productos", gridcolor="#E2E8F0"),
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig_hs, fig_zr

def grafico_importancia_features(metricas):
    imp = metricas.get("importance", {})
    if not imp: return None
    df_imp = pd.DataFrame({"Feature": list(imp.keys()), "Importancia": list(imp.values())})
    df_imp = df_imp.sort_values("Importancia", ascending=True).tail(10)
    fig = px.bar(df_imp, x="Importancia", y="Feature", orientation="h",
        color_discrete_sequence=[COLORS["primary"]], title="Top Features del Modelo ML")
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family":"Inter"})
    return fig


def grafico_waterfall_ahorro(pedidos_confirmados):
    """Waterfall chart mostrando ahorro acumulado por pedido/laboratorio."""
    if not pedidos_confirmados:
        return None
    labs = [p.get("laboratorio", "?") for p in pedidos_confirmados]
    fechas = [p.get("fecha", "") for p in pedidos_confirmados]
    ahorros = [p.get("ahorro_ofertas", 0) for p in pedidos_confirmados]
    labels = [f"{l}\n({f}) {i}" for i, (l, f) in enumerate(zip(labs, fechas))]
    acum = list(np.cumsum(ahorros))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=ahorros, name="Ahorro por Pedido",
        marker_color=[COLORS["primary"] if a > 0 else COLORS["danger"] for a in ahorros],
        text=[f"{a:,.0f}€" for a in ahorros], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Ahorro: %{y:,.2f}€<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=acum, name="Acumulado",
        line=dict(color=COLORS["success"], width=3),
        mode="lines+markers+text",
        text=[f"{v:,.0f}€" for v in acum], textposition="top center",
        textfont=dict(size=10, color=COLORS["success"]),
    ))
    fig.update_layout(
        title="💶 Ahorro Generado por Pedido", height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        xaxis=dict(title="Pedido (Laboratorio)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Ahorro (€)", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    return fig


def grafico_dinero_en_riesgo_donut(valor_zombie, valor_sobrestock, coste_oportunidad):
    """Donut chart: composicion instantanea del dinero en riesgo."""
    labels = ["Capital Zombie", "Sobrestock", "Venta Perdida (est.)"]
    values = [max(0, valor_zombie), max(0, valor_sobrestock), max(0, coste_oportunidad * 12)]
    total = sum(values)
    if total == 0:
        return None
    colors_donut = [COLORS["warning"], "#FFA726", COLORS["danger"]]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.55, marker=dict(colors=colors_donut),
        textinfo="label+percent", textposition="outside",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} EUR<br>%{percent}<extra></extra>",
    )])
    fig.update_layout(
        title="💰 Dinero en Riesgo", height=350,
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
        annotations=[dict(text=f"{total:,.0f}€", x=0.5, y=0.5, font_size=18,
                          font_family="Inter", font_weight="bold", showarrow=False)],
        showlegend=False,
    )
    return fig


# --- GRAFICOS NUEVOS DE AUDITORÍA ---
def grafico_distribucion_laboratorios(df_inv, tipo="Todos"):
    """Treemap o BarChart de valor inmovilizado por Laboratorio filtrado por tipo."""
    if df_inv.empty or COL_LAB not in df_inv.columns or COL_PVL not in df_inv.columns: return None
    df = df_inv.copy()
    df["Valor_Stock"] = (df[COL_STOCK] * df[COL_PVL]).fillna(0)
    
    if tipo != "Todos":
        df["Tipo_Prod"] = df.apply(lambda r: heuristica_tipo_producto(r.get(COL_CN,""), r.get(COL_MOLECULA if COL_MOLECULA in df.columns else COL_NOMBRE, "")), axis=1)
        df = df[df["Tipo_Prod"] == tipo]
        
    if df.empty: return None
    
    agg = df.groupby(COL_LAB)["Valor_Stock"].sum().reset_index()
    agg = agg[agg["Valor_Stock"] > 0].sort_values("Valor_Stock", ascending=False)
    if agg.empty: return None
    
    if len(agg) > 15:
        top15 = agg.iloc[:15].copy()
        resto = pd.DataFrame([{COL_LAB: "OTROS LABS", "Valor_Stock": agg.iloc[15:]["Valor_Stock"].sum()}])
        agg = pd.concat([top15, resto], ignore_index=True)
        
    fig = px.bar(agg.sort_values("Valor_Stock", ascending=True), 
                 x="Valor_Stock", y=COL_LAB, orientation="h",
                 title=f"Distribución de Inversión ({tipo})",
                 color_discrete_sequence=[COLORS["primary"]])
    fig.update_layout(height=450, margin={"t":40,"b":10,"l":10,"r":10}, xaxis_title="Euros Invertidos", yaxis_title="")
    return fig

def grafico_long_tail(df_inv, df_vm):
    """Pareto ancho de catalogo vs ventas"""
    if df_inv.empty or df_vm.empty: return None
    df = df_inv.merge(df_vm[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    
    df = df[(df[COL_STOCK]>0) | (df["Venta_Media_Mensual"]>0)]
    if df.empty: return None
    
    df = df.sort_values(by="Venta_Media_Mensual", ascending=False).reset_index(drop=True)
    total_ventas = df["Venta_Media_Mensual"].sum()
    if total_ventas == 0: return None
    
    df["Pct_Ventas_Acum"] = (df["Venta_Media_Mensual"].cumsum() / total_ventas) * 100
    df["Pct_Refs_Acum"] = ((df.index + 1) / len(df)) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Pct_Refs_Acum"], y=df["Pct_Ventas_Acum"], 
                             mode='lines', line=dict(color=COLORS["success"], width=3),
                             name='Ventas Acumuladas', fill='tozeroy'))
    fig.add_shape(type="line", x0=0, y0=80, x1=20, y1=80, line=dict(color="red", width=1, dash="dash"))
    fig.add_shape(type="line", x0=20, y0=0, x1=20, y1=80, line=dict(color="red", width=1, dash="dash"))
    
    fig.update_layout(title="Análisis de Long Tail (80/20)",
                      xaxis_title="% Referencias del Catálogo",
                      yaxis_title="% Ventas Generadas",
                      height=300, margin={"t":40,"b":40,"l":10,"r":10})
    return fig

def grafico_concentracion_riesgo(df_zombie, df_uvi, df_sobrestock):
    """Muestra qué laboratorios concentran el riesgo (combina los 3 dataframes)"""
    riesgos = []
    
    if not df_zombie.empty and "Valor_Inmovilizado" in df_zombie.columns:
        tz = df_zombie.groupby(COL_LAB)["Valor_Inmovilizado"].sum().reset_index()
        tz.columns = [COL_LAB, "Valor"]; tz["Tipo_Riesgo"] = "Zombie"
        riesgos.append(tz)
        
    if not df_uvi.empty and "Valor_Inmovilizado" in df_uvi.columns:
        tu = df_uvi.groupby(COL_LAB)["Valor_Inmovilizado"].sum().reset_index()
        tu.columns = [COL_LAB, "Valor"]; tu["Tipo_Riesgo"] = "UVI"
        riesgos.append(tu)
        
    if not df_sobrestock.empty and "Valor_Exceso" in df_sobrestock.columns:
        ts = df_sobrestock.groupby(COL_LAB)["Valor_Exceso"].sum().reset_index()
        ts.columns = [COL_LAB, "Valor"]; ts["Tipo_Riesgo"] = "Sobrestock"
        riesgos.append(ts)
        
    if not riesgos: return None
    
    df_r = pd.concat(riesgos, ignore_index=True)
    df_r = df_r[df_r["Valor"] > 0]
    if df_r.empty: return None
    
    top_labs = df_r.groupby(COL_LAB)["Valor"].sum().nlargest(10).index
    df_r_top = df_r[df_r[COL_LAB].isin(top_labs)]
    
    if df_r_top.empty: return None
    fig = px.bar(df_r_top, x="Valor", y=COL_LAB, color="Tipo_Riesgo", orientation="h",
                 title="Concentración de Capital en Riesgo (Top 10 Labs)",
                 color_discrete_map={"Zombie": COLORS["danger"], "UVI": COLORS["warning"], "Sobrestock": COLORS["primary"]})
    fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'}, height=350,
                      margin={"t":40,"b":10,"l":10,"r":10})
    return fig

def grafico_heatmap_cobertura(df_inv, df_vm):
    """Heatmap de meses de stock por molecula o lab principal para ver donde sobra y donde falta."""
    if df_inv.empty or df_vm.empty: return None
    col_agrupacion = COL_MOLECULA if COL_MOLECULA in df_inv.columns else COL_LAB
    if col_agrupacion not in df_inv.columns: return None
    
    df = df_inv.merge(df_vm[[COL_CN, "Venta_Media_Mensual"]], on=COL_CN, how="left")
    df["Venta_Media_Mensual"] = df["Venta_Media_Mensual"].fillna(0)
    
    agg = df.groupby(col_agrupacion).agg(Stock_Total=(COL_STOCK, "sum"), Venta_Media=("Venta_Media_Mensual", "sum")).reset_index()
    agg = agg[agg["Venta_Media"] > 2] 
    if agg.empty: return None
    
    agg["Meses_Cobertura"] = np.where(agg["Venta_Media"] > 0, agg["Stock_Total"] / agg["Venta_Media"], 0)
    agg["Meses_Cobertura"] = agg["Meses_Cobertura"].clip(upper=12).round(1)
    agg = agg.nlargest(25, "Venta_Media").sort_values("Meses_Cobertura", ascending=False)
    
    fig = px.imshow([agg["Meses_Cobertura"].values], 
                    x=agg[col_agrupacion].values, y=["Meses Stock"],
                    color_continuous_scale="RdYlGn_r", aspect="auto")
    fig.update_layout(title="Mapa de Calor: Cobertura de Stock (Meses)", height=250, margin={"t":40,"b":40,"l":10,"r":10})
    return fig

def grafico_estacionalidad_liquidez(df_stats):
    """Gráfico radar histórico para el índice de dependencia"""
    if df_stats.empty: return None
    fig = px.line_polar(df_stats, r="Pct_Ventas", theta="Mes_Nombre", line_close=True,
                        title="Radar de Riesgo de Liquidez Anual (Estacionalidad)")
    fig.update_traces(fill='toself', line_color=COLORS["primary"])
    fig.update_layout(height=350, margin={"t":40,"b":20,"l":20,"r":20})
    return fig


def grafico_roi_laboratorios(df_roi, top_n=15):
    """Barras horizontales de ROI por laboratorio, con colores verde/rojo."""
    if df_roi.empty:
        return None
    df_top = df_roi.head(top_n).sort_values("ROI", ascending=True)
    media_roi = df_roi["ROI"].mean()
    colores = [COLORS["success"] if r >= media_roi else COLORS["danger"] for r in df_top["ROI"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_top["ROI"], y=df_top[COL_LAB], orientation="h",
        marker_color=colores,
        text=[f"ROI: {r:.1f}x | {n} prods" for r, n in zip(df_top["ROI"], df_top["N_Productos"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ROI: %{x:.2f}x<br>Stock: %{customdata[0]:,.0f}€<br>Venta Anual: %{customdata[1]:,.0f}€<extra></extra>",
        customdata=list(zip(df_top["Stock_EUR"], df_top["Venta_Anual_EUR"])),
    ))
    fig.add_vline(x=media_roi, line_dash="dash", line_color=COLORS["muted"],
        annotation_text=f"Media: {media_roi:.1f}x", annotation_position="top right")
    fig.update_layout(
        title="📊 ROI por Laboratorio (Venta Anual / Stock)", height=max(300, top_n * 28),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        xaxis=dict(title="ROI (veces)", gridcolor="#E2E8F0"),
        yaxis=dict(title=""),
    )
    return fig


# Cache
def obtener_ventas_media(meses_cobertura=2):
    df_hist = st.session_state.get("historico")
    if df_hist is None: return pd.DataFrame()
    current_hash = hash((pd.util.hash_pandas_object(df_hist).sum(), meses_cobertura))
    if "ventas_media_cache" not in st.session_state or st.session_state.get("historico_hash") != current_hash:
        st.session_state["ventas_media_cache"] = calcular_ventas_mensuales(df_hist, meses_cobertura)
        st.session_state["historico_hash"] = current_hash
    return st.session_state["ventas_media_cache"]

def obtener_modelo_cacheado():
    if "modelo_ml_cache" not in st.session_state:
        model, met, rmse = cargar_modelo_farmacia()
        st.session_state["modelo_ml_cache"] = (model, met, rmse)
    return st.session_state["modelo_ml_cache"]

def invalidar_cache_modelo():
    st.session_state.pop("modelo_ml_cache", None)

def necesita_reentrenamiento():
    """Detecta si el historico ha cambiado desde el ultimo entrenamiento."""
    _, met, _ = obtener_modelo_cacheado()
    if met is None:
        return False
    n_hist_modelo = met.get("n_registros_historico", 0)
    df_hist = st.session_state.get("historico")
    if df_hist is None:
        return False
    return len(df_hist) != n_hist_modelo


MARGEN_SEGURIDAD_DIAS = 14  # 2 semanas antes de limite


# ===========================================================================
# TORRE DE CONTROL — Backend: Pedidos confirmados, Red, Ventanas, Oportunidades
# ===========================================================================

# --- Red ---
def cargar_red_config():
    path = BASE_DIR / "red_config.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"farmacias_activas": [], "fecha_creacion": datetime.now().strftime("%Y-%m-%d")}

def guardar_red_config(config):
    path = BASE_DIR / "red_config.json"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

def cargar_historico_compras_conjuntas():
    path = BASE_DIR / "historico_compras_conjuntas.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def guardar_historico_compras_conjuntas(historico):
    path = BASE_DIR / "historico_compras_conjuntas.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2, default=str)


# --- Pedidos confirmados (por farmacia) ---
def cargar_pedidos_confirmados(nombre_farmacia=None):
    if nombre_farmacia:
        path = BASE_DIR / nombre_farmacia / "historico_pedidos.json"
    else:
        ruta = ruta_farmacia_activa()
        if ruta is None:
            return []
        path = ruta / "historico_pedidos.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def guardar_pedido_confirmado(registro, nombre_farmacia=None):
    if nombre_farmacia:
        path = BASE_DIR / nombre_farmacia / "historico_pedidos.json"
    else:
        ruta = ruta_farmacia_activa()
        if ruta is None:
            return
        path = ruta / "historico_pedidos.json"
    historico = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                historico = json.load(f)
        except Exception:
            pass
    historico.append(registro)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2, default=str)


def registrar_pedido_confirmado(df_pedido, laboratorio, modo, meses_cobertura, df_ventas_media):
    """Registra un pedido confirmado con datos para calcular ventana."""
    if df_pedido is None or df_pedido.empty:
        return None

    # Calcular venta media diaria por producto
    vm_map = {}
    if df_ventas_media is not None and not df_ventas_media.empty:
        vm_map = df_ventas_media.set_index(COL_CN)["Venta_Media_Mensual"].to_dict()

    productos = []
    for _, row in df_pedido.iterrows():
        cn = row.get(COL_CN, "")
        cant = int(row.get("Cantidad_A_Pedir", 0))
        pvl = float(row.get("Precio_Unitario", row.get(COL_PVL, 0)))
        molecula = str(row.get(COL_MOLECULA, ""))
        dto = float(row.get("Descuento_Aplicado", 0))
        vmd = vm_map.get(cn, 0) / 30.44  # mensual a diaria (media precisa)
        dias_cob = round(cant / vmd, 1) if vmd > 0 else 999
        productos.append({
            "cn": cn, "nombre": str(row.get(COL_NOMBRE, ""))[:50],
            "molecula": molecula,
            "cantidad": cant, "pvl": round(pvl, 4),
            "descuento": round(dto, 4),
            "coste": round(float(row.get("Coste_Con_Dto", 0)), 2),
            "venta_media_diaria": round(vmd, 2), "dias_cobertura": dias_cob,
        })

    # Dias cobertura del pedido = minimo de todos los productos con venta
    prods_con_venta = [p for p in productos if p["venta_media_diaria"] > 0]
    if prods_con_venta:
        dias_cob_pedido = min(p["dias_cobertura"] for p in prods_con_venta)
        cuello_botella = min(prods_con_venta, key=lambda p: p["dias_cobertura"])["nombre"]
    else:
        dias_cob_pedido = (meses_cobertura or 2) * 30
        cuello_botella = ""

    registro = {
        "fecha": datetime.now().strftime("%Y-%m-%d"),
        "laboratorio": laboratorio,
        "modo": modo,
        "meses_cobertura": meses_cobertura if modo == "cobertura" else None,
        "dias_cobertura_estimados": round(dias_cob_pedido),
        "producto_cuello_botella": cuello_botella,
        "coste_total": round(float(df_pedido["Coste_Con_Dto"].sum()), 2),
        "ahorro_ofertas": round(float(df_pedido["Ahorro"].sum()), 2),
        "n_lineas": len(df_pedido),
        "unidades_total": int(df_pedido["Cantidad_A_Pedir"].sum()),
        "productos": productos,
    }
    guardar_pedido_confirmado(registro)
    return registro


# --- Ventana de pedido ---
def calcular_ventana_pedido(nombre_farmacia, laboratorio):
    """Calcula semana optima y limite del proximo pedido."""
    pedidos = cargar_pedidos_confirmados(nombre_farmacia)
    # Filtrar por laboratorio
    pedidos_lab = [p for p in pedidos if p.get("laboratorio", "").lower() == laboratorio.lower()]
    if not pedidos_lab:
        return None

    ultimo = pedidos_lab[-1]
    fecha_pedido = datetime.strptime(ultimo["fecha"], "%Y-%m-%d").date()
    dias_cob = ultimo.get("dias_cobertura_estimados", 60)

    fecha_limite = fecha_pedido + timedelta(days=dias_cob)
    fecha_optima = fecha_limite - timedelta(days=MARGEN_SEGURIDAD_DIAS)

    return {
        "farmacia": nombre_farmacia,
        "laboratorio": laboratorio,
        "fecha_ultimo_pedido": fecha_pedido,
        "fecha_optima": max(fecha_optima, date.today()),
        "fecha_limite": max(fecha_limite, date.today()),
        "dias_hasta_optima": (max(fecha_optima, date.today()) - date.today()).days,
        "dias_hasta_limite": (max(fecha_limite, date.today()) - date.today()).days,
        "ultimo_pedido": ultimo,
    }


def obtener_ventanas_red(laboratorio):
    """Calcula ventanas de todas las farmacias de la red para un laboratorio."""
    red = cargar_red_config()
    ventanas = []
    for farm in red.get("farmacias_activas", []):
        v = calcular_ventana_pedido(farm, laboratorio)
        if v:
            ventanas.append(v)
    return ventanas


# --- Deteccion de solapamientos ---
def detectar_solapamientos(ventanas):
    """Encuentra pares/grupos de farmacias con ventanas solapadas."""
    if len(ventanas) < 2:
        return []
    solapamientos = []
    for i in range(len(ventanas)):
        for j in range(i + 1, len(ventanas)):
            vi, vj = ventanas[i], ventanas[j]
            # Solapamiento: el optimo de uno cae dentro del rango del otro
            inicio = max(vi["fecha_optima"], vj["fecha_optima"])
            fin = min(vi["fecha_limite"], vj["fecha_limite"])
            if inicio <= fin:
                solapamientos.append({
                    "farmacias": [vi["farmacia"], vj["farmacia"]],
                    "inicio_solape": inicio,
                    "fin_solape": fin,
                    "dias_solape": (fin - inicio).days,
                    "ventanas": [vi, vj],
                })
    return solapamientos


# --- Optimizacion de timing (Mejora #1) ---
def optimizar_timing_red(ventanas):
    """Sugiere mover pedidos dentro de ventana segura para crear solapamientos."""
    if len(ventanas) < 2:
        return []
    oportunidades = []
    for i in range(len(ventanas)):
        for j in range(i + 1, len(ventanas)):
            vi, vj = ventanas[i], ventanas[j]
            # Ya solapan naturalmente?
            inicio_nat = max(vi["fecha_optima"], vj["fecha_optima"])
            fin_nat = min(vi["fecha_limite"], vj["fecha_limite"])
            if inicio_nat <= fin_nat:
                # Solapamiento natural
                oportunidades.append({
                    "tipo": "natural",
                    "farmacias": [vi["farmacia"], vj["farmacia"]],
                    "fecha_sugerida": inicio_nat,
                    "fin_ventana": fin_nat,
                    "dias_disponibles": (fin_nat - inicio_nat).days,
                    "ajuste_necesario": None,
                    "ventanas": [vi, vj],
                })
            else:
                # Puede uno esperar/adelantar para solapar?
                # Caso 1: Farm i espera hasta optima de j (si j optima <= i limite)
                if vj["fecha_optima"] <= vi["fecha_limite"] and vj["fecha_optima"] >= vi["fecha_optima"]:
                    dias_disp = (vi["fecha_limite"] - vj["fecha_optima"]).days
                    if dias_disp > 0:
                        oportunidades.append({
                            "tipo": "optimizado",
                            "farmacias": [vi["farmacia"], vj["farmacia"]],
                            "fecha_sugerida": vj["fecha_optima"],
                            "fin_ventana": min(vi["fecha_limite"], vj["fecha_limite"]),
                            "dias_disponibles": dias_disp,
                            "ajuste_necesario": f"{vi['farmacia']} espera {(vj['fecha_optima'] - vi['fecha_optima']).days} dias",
                            "ventanas": [vi, vj],
                        })
                # Caso 2: Farm j espera hasta optima de i (si i optima <= j limite)
                elif vi["fecha_optima"] <= vj["fecha_limite"] and vi["fecha_optima"] >= vj["fecha_optima"]:
                    dias_disp = (vj["fecha_limite"] - vi["fecha_optima"]).days
                    if dias_disp > 0:
                        oportunidades.append({
                            "tipo": "optimizado",
                            "farmacias": [vj["farmacia"], vi["farmacia"]],
                            "fecha_sugerida": vi["fecha_optima"],
                            "fin_ventana": min(vi["fecha_limite"], vj["fecha_limite"]),
                            "dias_disponibles": dias_disp,
                            "ajuste_necesario": f"{vj['farmacia']} espera {(vi['fecha_optima'] - vj['fecha_optima']).days} dias",
                            "ventanas": [vi, vj],
                        })
    oportunidades.sort(key=lambda o: o["dias_disponibles"], reverse=True)
    return oportunidades


# --- Simulacion pedido conjunto (F5) ---
def simular_pedido_conjunto(farmacias_pedidos, df_ofertas):
    """Toma pedidos individuales, suma cantidades y recalcula tiers con volumen conjunto.
    farmacias_pedidos: list of (nombre_farmacia, df_pedido)
    """
    if not farmacias_pedidos or not any(df is not None and not df.empty for _, df in farmacias_pedidos):
        return None

    # Combinar todos los pedidos
    dfs = []
    for farm_name, df_p in farmacias_pedidos:
        if df_p is not None and not df_p.empty:
            dfx = df_p.copy()
            dfx["_farmacia"] = farm_name
            dfs.append(dfx)
    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    # Sumar cantidades por producto
    agg_cols = {
        "Cantidad_A_Pedir": "sum",
        "Precio_Unitario": "first",
    }
    for c in [COL_NOMBRE, COL_MOLECULA, COL_LAB]:
        if c in df_all.columns:
            agg_cols[c] = "first"

    df_conjunto = df_all.groupby(COL_CN).agg(agg_cols).reset_index()

    # Reaplicar ofertas con volumen conjunto
    df_conjunto = aplicar_ofertas_indexado(df_conjunto, df_ofertas)
    df_conjunto["Coste_Sin_Dto"] = df_conjunto["Cantidad_A_Pedir"] * df_conjunto["Precio_Unitario"]
    df_conjunto["Coste_Con_Dto"] = df_conjunto["Coste_Sin_Dto"] * (1 - df_conjunto["Descuento_Aplicado"])
    df_conjunto["Ahorro"] = df_conjunto["Coste_Sin_Dto"] - df_conjunto["Coste_Con_Dto"]

    # Calcular ahorro por farmacia (proporcional al coste sin descuento u original)
    coste_conjunto_total = df_conjunto["Coste_Con_Dto"].sum()
    coste_individual_total = sum(df_p["Coste_Con_Dto"].sum() for _, df_p in farmacias_pedidos if df_p is not None and not df_p.empty and "Coste_Con_Dto" in df_p.columns)
    resumen_farmacias = []
    for farm_name, df_p in farmacias_pedidos:
        if df_p is None or df_p.empty:
            continue
        coste_individual = df_p["Coste_Con_Dto"].sum() if "Coste_Con_Dto" in df_p.columns else 0
        uds_farm = df_p["Cantidad_A_Pedir"].sum()
        proporcion = coste_individual / coste_individual_total if coste_individual_total > 0 else 0
        coste_farm_conjunto = coste_conjunto_total * proporcion
        ahorro_farm = coste_individual - coste_farm_conjunto
        resumen_farmacias.append({
            "farmacia": farm_name,
            "coste_individual": round(coste_individual, 2),
            "coste_conjunto": round(coste_farm_conjunto, 2),
            "ahorro": round(max(0, ahorro_farm), 2),
            "unidades": int(uds_farm),
            "proporcion": round(proporcion * 100, 1),
        })

    # Distancia al siguiente tier
    tier_info = None
    if df_ofertas is not None and not df_ofertas.empty:
        indice = construir_indice_ofertas(df_ofertas)
        for _, row in df_conjunto.iterrows():
            oferta = buscar_oferta_por_indice(
                str(row.get(COL_NOMBRE, "")), str(row.get(COL_MOLECULA, "")), indice, df_ofertas)
            if oferta is not None:
                tiers = oferta.get("Tiers", [])
                if isinstance(tiers, str):
                    try: tiers = json.loads(tiers)
                    except: tiers = []
                for t in tiers:
                    if int(row["Cantidad_A_Pedir"]) < t["min"]:
                        faltan = t["min"] - int(row["Cantidad_A_Pedir"])
                        tier_info = {
                            "producto": str(row.get(COL_NOMBRE, "")),
                            "tier_actual": row.get("Tier_Aplicado", "Sin oferta"),
                            "siguiente_tier_min": t["min"],
                            "siguiente_tier_dto": f"{t['dto']*100:.0f}%",
                            "faltan_uds": faltan,
                        }
                        break

    return {
        "df_conjunto": df_conjunto,
        "coste_total_conjunto": round(coste_conjunto_total, 2),
        "ahorro_total": round(sum(r["ahorro"] for r in resumen_farmacias), 2),
        "resumen_farmacias": resumen_farmacias,
        "tier_info": tier_info,
    }


# --- Coste de no actuar (Mejora #2) ---
def cargar_ahorro_perdido(nombre_farmacia=None):
    if nombre_farmacia:
        path = BASE_DIR / nombre_farmacia / "ahorro_perdido.json"
    else:
        ruta = ruta_farmacia_activa()
        if ruta is None:
            return {"oportunidades": [], "total_perdido": 0}
        path = ruta / "ahorro_perdido.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"oportunidades": [], "total_perdido": 0}

def registrar_ahorro_perdido(nombre_farmacia, ahorro, laboratorio, descripcion):
    path = BASE_DIR / nombre_farmacia / "ahorro_perdido.json"
    data = cargar_ahorro_perdido(nombre_farmacia)
    data["oportunidades"].append({
        "fecha": datetime.now().strftime("%Y-%m-%d"),
        "ahorro_perdido": round(ahorro, 2),
        "laboratorio": laboratorio,
        "descripcion": descripcion,
    })
    data["total_perdido"] = round(sum(o["ahorro_perdido"] for o in data["oportunidades"]), 2)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# --- Notificaciones ---
def generar_notificaciones():
    """Genera alertas de oportunidades de compra conjunta."""
    red = cargar_red_config()
    farmacias = red.get("farmacias_activas", [])
    if len(farmacias) < 2:
        return []

    # Recoger todos los laboratorios con pedidos
    labs = set()
    for farm in farmacias:
        pedidos = cargar_pedidos_confirmados(farm)
        for p in pedidos:
            labs.add(p.get("laboratorio", ""))

    alertas = []
    for lab in labs:
        if not lab:
            continue
        ventanas = obtener_ventanas_red(lab)
        if len(ventanas) < 2:
            continue
        oportunidades = optimizar_timing_red(ventanas)
        for op in oportunidades:
            if op["dias_disponibles"] > 0:
                alertas.append({
                    "laboratorio": lab,
                    "tipo": op["tipo"],
                    "farmacias": op["farmacias"],
                    "fecha_sugerida": op["fecha_sugerida"],
                    "dias_disponibles": op["dias_disponibles"],
                    "ajuste": op.get("ajuste_necesario"),
                })
    return alertas


# --- Registrar compra conjunta ---
def registrar_compra_conjunta(farmacias, laboratorio, ahorro_total, resumen_farmacias):
    historico = cargar_historico_compras_conjuntas()
    historico.append({
        "fecha": datetime.now().strftime("%Y-%m-%d"),
        "laboratorio": laboratorio,
        "farmacias": farmacias,
        "ahorro_total": round(ahorro_total, 2),
        "detalle": resumen_farmacias,
    })
    guardar_historico_compras_conjuntas(historico)


# ===========================================================================
# MODULO 0: SELECTOR DE FARMACIA
# ===========================================================================
def modulo_selector_farmacia():
    st.markdown("### \U0001f3e5 Selecciona la Farmacia")
    farmacias = obtener_farmacias_disponibles()
    col_sel, col_new = st.columns([2, 1])
    with col_sel:
        if farmacias:
            seleccion = st.selectbox("Farmacias disponibles:", ["-- Selecciona --"] + farmacias, key="sel_farmacia")
            if seleccion != "-- Selecciona --":
                if st.button("\u2705 Abrir Farmacia", type="primary", use_container_width=True):
                    st.session_state["farmacia_activa"] = seleccion
                    for k in ["inventario","historico","ofertas_normalizadas","ventas_media_cache",
                              "historico_hash","pedido_generado","modelo_ml_cache","ofertas_raw",
                              "tabla_maestra_atc","temperatura_historica","alertas_red_cache"]:
                        st.session_state.pop(k, None)
                    st.rerun()
        else:
            st.info("No hay farmacias creadas.")
    with col_new:
        st.markdown("**Crear nueva:**")
        nn = st.text_input("Nombre:", key="nueva_farmacia", placeholder="Ej: Farmacia Lopez")
        if st.button("\U0001f4be Crear", use_container_width=True) and nn.strip():
            nl = crear_farmacia(nn)
            st.session_state["farmacia_activa"] = nl
            st.rerun()


# ===========================================================================
# MODULO 1: CONFIGURACION (sin meses_cobertura, con nuevos campos)
# ===========================================================================
def modulo_configuracion():
    st.markdown("### \u2699\ufe0f Centro de Control")

    # --- A. Archivos ---
    st.markdown("#### A. Archivos Base")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        file_inv = st.file_uploader("\U0001f4e6 INVENTARIO", type=["xlsx","xls","csv"], key="up_inv")
        st.markdown('</div>', unsafe_allow_html=True)
        if file_inv:
            try:
                df_inv = pd.read_csv(file_inv) if file_inv.name.endswith(".csv") else pd.read_excel(file_inv)
                df_inv, inf = validar_y_renombrar_columnas(df_inv, [COL_CN, COL_STOCK, COL_PVL, COL_LAB, COL_NOMBRE, COL_MOLECULA])
                st.session_state["inventario"] = df_inv
                guardar_dataframe_farmacia("inventario.parquet", df_inv)
                for col, found in inf.items(): st.markdown(f"\u2705 `{col}`" if found else f"\u274c `{col}` no encontrada")
                st.info(f"{len(df_inv)} productos cargados")
                # Snapshot de auditoria automatico
                df_vm_snap = obtener_ventas_media(meses_cobertura=2) if "historico" in st.session_state else pd.DataFrame()
                if not df_vm_snap.empty:
                    snap = registrar_snapshot_auditoria(df_inv, df_vm_snap)
                    st.caption(f"\U0001f4f8 Snapshot guardado: {snap['n_zombies']} zombies, {snap['n_roturas']} roturas")
            except Exception as e: st.error(f"Error: {e}")
    with c2:
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        file_v = st.file_uploader("\U0001f4c8 HISTORICO VENTAS (36 meses)", type=["xlsx","xls","csv"], key="up_ven")
        st.markdown('</div>', unsafe_allow_html=True)
        if file_v:
            try:
                df_v = pd.read_csv(file_v) if file_v.name.endswith(".csv") else pd.read_excel(file_v)
                df_v, inf = validar_y_renombrar_columnas(df_v, [COL_CN, COL_VENTAS, COL_FECHA])
                st.session_state["historico"] = df_v
                guardar_dataframe_farmacia("historico.parquet", df_v)
                st.session_state.pop("ventas_media_cache", None); st.session_state.pop("historico_hash", None)
                for col, found in inf.items(): st.markdown(f"\u2705 `{col}`" if found else f"\u274c `{col}` no encontrada")
                if COL_FECHA in df_v.columns:
                    df_v[COL_FECHA] = pd.to_datetime(df_v[COL_FECHA], errors="coerce", dayfirst=True)
                    anios = df_v[COL_FECHA].dropna().dt.year.nunique()
                    st.info(f"{len(df_v)} registros | {anios} anio(s)")
                else: st.info(f"{len(df_v)} registros")
            except Exception as e: st.error(f"Error: {e}")

    # --- B. Ofertas Tiers Dinamicos ---
    st.divider()
    st.markdown("#### B. Ofertas del Laboratorio")
    file_of = st.file_uploader("\U0001f3f7\ufe0f OFERTAS", type=["xlsx","xls","csv"], key="up_of")
    if file_of:
        try:
            df_of = pd.read_csv(file_of) if file_of.name.endswith(".csv") else pd.read_excel(file_of)
            st.session_state["ofertas_raw"] = df_of
        except Exception as e: st.error(f"Error: {e}")
    df_of_raw = st.session_state.get("ofertas_raw")
    if df_of_raw is not None and not df_of_raw.empty:
        st.dataframe(df_of_raw.head(5), use_container_width=True, hide_index=True)
        cols_d = list(df_of_raw.columns); na = "-- No aplica --"; opts = [na] + cols_d
        mn = st.selectbox("Columna Nombre/Molecula:", opts, key="of_mn")
        st.markdown("**Tramos:**")
        n_t = int(st.number_input("Numero de tiers:", 1, 10, 2, key="n_tiers"))
        mt_list = []; cols_t = st.columns(min(int(n_t), 5))
        for i in range(int(n_t)):
            with cols_t[i % len(cols_t)]:
                st.markdown(f"**T{i+1}:**")
                cc = st.selectbox(f"Cant T{i+1}:", opts, key=f"oc_{i}")
                cd = st.selectbox(f"Dto T{i+1}:", opts, key=f"od_{i}")
                if cc != na and cd != na: mt_list.append({"col_cantidad": cc, "col_descuento": cd})
        if st.button("\u2728 Procesar Ofertas", type="primary"):
            if mn == na: st.error("Mapea Nombre/Molecula.")
            elif not mt_list: st.error("Mapea al menos un tramo.")
            else:
                df_n = normalizar_ofertas_dinamico(df_of_raw, mn, mt_list)
                st.session_state["ofertas_normalizadas"] = df_n
                guardar_dataframe_farmacia("ofertas_normalizadas.parquet", df_n)
                st.success(f"\u2705 {len(df_n)} ofertas procesadas.")

    # --- C. Reglas de Surtido ---
    st.divider()
    st.markdown("#### C. Reglas de Surtido")
    st.caption("Formato: Molecula,Dosis,Cantidad. Sin reglas = optimizacion libre.")
    reglas = cargar_reglas_surtido()
    df_reg = pd.DataFrame(reglas) if reglas else pd.DataFrame(columns=["presentacion","min_labs","labs_obligatorios"])
    df_reg_e = st.data_editor(df_reg, num_rows="dynamic", use_container_width=True, key="ed_reg")
    if st.button("\U0001f4be Guardar Reglas Surtido"):
        guardar_reglas_surtido(df_reg_e.dropna(subset=["presentacion"]).to_dict("records"))
        st.success("\u2705 Guardadas.")

    # --- D. Productos Protegidos (Excel + editor) ---
    st.divider()
    st.markdown("#### D. Productos Protegidos")
    st.caption("Carga por Excel o edita manualmente. Formato: codigo_nacional, nombre, stock_minimo, motivo")

    file_prot = st.file_uploader("\U0001f4c4 Cargar desde Excel:", type=["xlsx","xls","csv"], key="up_prot")
    if file_prot:
        try:
            df_prot_up = pd.read_csv(file_prot) if file_prot.name.endswith(".csv") else pd.read_excel(file_prot)
            df_prot_up, _ = validar_y_renombrar_columnas(df_prot_up, ["codigo_nacional", "nombre", "stock_minimo", "motivo"])
            if "codigo_nacional" in df_prot_up.columns:
                nuevos = df_prot_up.dropna(subset=["codigo_nacional"]).to_dict("records")
                guardar_productos_protegidos(nuevos)
                st.success(f"\u2705 {len(nuevos)} productos protegidos cargados desde Excel.")
            else:
                st.error("El Excel debe tener columna 'codigo_nacional'.")
        except Exception as e: st.error(f"Error: {e}")

    prot = cargar_productos_protegidos()
    df_pr = pd.DataFrame(prot) if prot else pd.DataFrame(columns=["codigo_nacional","nombre","stock_minimo","motivo"])
    df_pr_e = st.data_editor(df_pr, num_rows="dynamic", use_container_width=True, key="ed_prot")
    if st.button("\U0001f4be Guardar Protegidos"):
        guardar_productos_protegidos(df_pr_e.dropna(subset=["codigo_nacional"]).to_dict("records"))
        st.success("\u2705 Guardados.")

    # --- E. Perfil de la Farmacia (3 epis) ---
    st.divider()
    st.markdown("#### E. Perfil de la Farmacia")
    st.caption("Configura una vez. Afecta a las predicciones ML.")
    perfil = cargar_perfil_farmacia()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Proximidad:**")
        perfil["centro_salud"] = st.checkbox("Centro de Salud", perfil.get("centro_salud", False), key="pf_cs")
        perfil["residencia"] = st.checkbox("Residencia Mayores", perfil.get("residencia", False), key="pf_res")
        perfil["colegio"] = st.checkbox("Colegio", perfil.get("colegio", False), key="pf_col")
        perfil["zona_turistica"] = st.checkbox("Zona Turistica", perfil.get("zona_turistica", False), key="pf_tur")
        perfil["zona_rural"] = st.checkbox("Zona Rural", perfil.get("zona_rural", False), key="pf_rur")
    with c2:
        st.markdown("**Nivel Epidemiologico:**")
        epi_labels = {0: "0 \u2014 Normal", 1: "1 \u2014 Alerta leve", 2: "2 \u2014 Epidemia activa"}
        perfil["epi_gripe"] = st.selectbox("\U0001f912 Gripe / Respiratorio:", [0,1,2],
            index=perfil.get("epi_gripe", 0), key="epi_g", format_func=lambda x: epi_labels[x])
        perfil["epi_alergias"] = st.selectbox("\U0001f33c Alergias / Polen:", [0,1,2],
            index=perfil.get("epi_alergias", 0), key="epi_a", format_func=lambda x: epi_labels[x])
        perfil["epi_covid"] = st.selectbox("\U0001f9eb COVID / Otros:", [0,1,2],
            index=perfil.get("epi_covid", 0), key="epi_c", format_func=lambda x: epi_labels[x])
        st.caption("Ref: svge.isciii.es (gripe), polenes.com (alergias)")
        st.markdown("**Zona Climatica:**")
        zonas_clima = ["Mediterraneo", "Continental", "Atlantico"]
        zona_actual = perfil.get("zona_climatica", "mediterraneo").capitalize()
        if zona_actual not in zonas_clima:
            zona_actual = "Mediterraneo"
        perfil["zona_climatica"] = st.selectbox("\U0001f321\ufe0f Clima:", zonas_clima,
            index=zonas_clima.index(zona_actual), key="zona_cli").lower()
    if st.button("\U0001f4be Guardar Perfil"):
        guardar_perfil_farmacia(perfil)
        st.success("\u2705 Perfil guardado.")

    # --- F. Calendario (3 capas) ---
    st.divider()
    st.markdown("#### F. Calendario de la Farmacia")
    cal = cargar_calendario_farmacia()
    horario = cal.get("horario", {})

    st.markdown("**Horario semanal:**")
    dias_es = ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"]
    dias_labels = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"]

    cols_h = st.columns(7)
    for i, (dia, label) in enumerate(zip(dias_es, dias_labels)):
        cfg = horario.get(dia, {"abre": i < 6, "apertura": "09:30", "cierre": "21:00"})
        with cols_h[i]:
            st.markdown(f"**{label}**")
            cfg["abre"] = st.checkbox("Abre", cfg.get("abre", i < 6), key=f"h_abre_{dia}")
            if cfg["abre"]:
                cfg["apertura"] = st.text_input("Desde:", cfg.get("apertura", "09:30"), key=f"h_ap_{dia}")
                cfg["cierre"] = st.text_input("Hasta:", cfg.get("cierre", "21:00"), key=f"h_ci_{dia}")
            horario[dia] = cfg

    st.markdown("**Festivos** (uno por linea, formato DD/MM/AAAA):")
    festivos_str = "\n".join(cal.get("festivos", []))
    nuevos_festivos = st.text_area("Festivos:", festivos_str, height=80, key="cal_fest")

    st.markdown("**Excepciones: dias que ABRE aunque sea festivo** (guardias, etc.):")
    excepciones_str = "\n".join(cal.get("excepciones_abiertas", []))
    nuevas_excepciones = st.text_area("Excepciones:", excepciones_str, height=60, key="cal_exc")

    if st.button("\U0001f4be Guardar Calendario"):
        cal_nuevo = {
            "horario": horario,
            "festivos": [d.strip() for d in nuevos_festivos.split("\n") if d.strip()],
            "excepciones_abiertas": [d.strip() for d in nuevas_excepciones.split("\n") if d.strip()],
        }
        guardar_calendario_farmacia(cal_nuevo)
        st.success("\u2705 Calendario guardado.")

    # --- G. Datos Complementarios (ML) ---
    st.divider()
    st.markdown("#### G. Datos Complementarios (ML)")
    st.caption("Opcionales. Mejoran la precision del modelo si se proporcionan.")
    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown("**Tabla Maestra ATC** (Excel: Codigo_Nacional + Grupo_ATC)")
        file_atc = st.file_uploader("\U0001f9ec Tabla ATC", type=["xlsx","xls","csv"], key="up_atc")
        if file_atc:
            try:
                if file_atc.name.endswith(".csv"):
                    df_atc = pd.read_csv(file_atc, dtype=str)
                else:
                    df_atc = pd.read_excel(file_atc, dtype=str)
                st.session_state["tabla_maestra_atc"] = df_atc
                # Persistir en carpeta de farmacia
                ruta_f = ruta_farmacia_activa()
                if ruta_f:
                    df_atc.to_csv(ruta_f / "tabla_atc.csv", index=False)
                st.success(f"\u2705 Tabla ATC: {len(df_atc)} registros | Columnas: {', '.join(df_atc.columns[:4])}")
            except Exception as e:
                st.error(f"Error leyendo tabla ATC: {e}")
        elif "tabla_maestra_atc" not in st.session_state:
            # Intentar cargar de disco
            ruta_f = ruta_farmacia_activa()
            if ruta_f and (ruta_f / "tabla_atc.csv").exists():
                st.session_state["tabla_maestra_atc"] = pd.read_csv(ruta_f / "tabla_atc.csv", dtype=str)
        if "tabla_maestra_atc" in st.session_state:
            st.info(f"\u2705 Tabla ATC cargada ({len(st.session_state['tabla_maestra_atc'])} registros)")
        else:
            st.caption("Sin tabla ATC se usa proxy por molecula (funciona pero es menos preciso).")
    with cg2:
        st.markdown("**Temperatura Historica** (CSV: Anio, Mes, Temp_Media)")
        file_temp = st.file_uploader("\U0001f321\ufe0f Temperatura", type=["csv","xlsx"], key="up_temp")
        if file_temp:
            try:
                if file_temp.name.endswith(".csv"):
                    df_temp = pd.read_csv(file_temp)
                else:
                    df_temp = pd.read_excel(file_temp)
                for c in ["Anio", "Mes", "Temp_Media"]:
                    df_temp[c] = pd.to_numeric(df_temp[c], errors="coerce")
                df_temp = df_temp.dropna(subset=["Anio", "Mes", "Temp_Media"])
                st.session_state["temperatura_historica"] = df_temp
                # Persistir en carpeta de farmacia
                ruta_f = ruta_farmacia_activa()
                if ruta_f:
                    df_temp.to_csv(ruta_f / "temperatura.csv", index=False)
                st.success(f"\u2705 Temperatura: {len(df_temp)} meses cargados")
            except Exception as e:
                st.error(f"Error leyendo temperatura: {e}")
        elif "temperatura_historica" not in st.session_state:
            # Intentar cargar de disco
            ruta_f = ruta_farmacia_activa()
            if ruta_f and (ruta_f / "temperatura.csv").exists():
                df_t = pd.read_csv(ruta_f / "temperatura.csv")
                for c in ["Anio", "Mes", "Temp_Media"]:
                    df_t[c] = pd.to_numeric(df_t[c], errors="coerce")
                st.session_state["temperatura_historica"] = df_t.dropna(subset=["Anio","Mes","Temp_Media"])
        if "temperatura_historica" in st.session_state:
            st.info(f"\u2705 Temperatura cargada ({len(st.session_state['temperatura_historica'])} meses)")
        else:
            st.caption(f"Sin CSV se usa climatologia de la zona ({perfil.get('zona_climatica', 'mediterraneo').capitalize()}).")

    # --- H. Motor de Prediccion ML ---
    st.divider()
    st.markdown("#### H. Motor de Prediccion ML")
    if not ML_AVAILABLE:
        st.warning("Instala xgboost y scikit-learn: `pip install xgboost scikit-learn joblib`")
    else:
        model_ok, met, _ = obtener_modelo_cacheado()
        if model_ok and met:
            fecha_ent = met.get("fecha_entrenamiento", "?")
            st.success(f"\u2705 Modelo entrenado | RMSE: {met.get('rmse','?')} | R\u00b2: {met.get('r2','?')} | Fecha: {fecha_ent}")
            if necesita_reentrenamiento():
                st.warning("\u26a0\ufe0f El historico ha cambiado desde el ultimo entrenamiento. Reentrena para mejores resultados.")
        else:
            st.info("No hay modelo entrenado para esta farmacia.")

        if "inventario" in st.session_state and "historico" in st.session_state:
            if st.button("\U0001f9e0 Entrenar Modelo ML", type="primary", use_container_width=True):
                with st.spinner("Entrenando modelo..."):
                    df_hist_imp = imputar_stockouts(st.session_state["historico"], st.session_state["inventario"])
                    perfil_f = cargar_perfil_farmacia()
                    cal_f = cargar_calendario_farmacia()
                    ofertas_n = st.session_state.get("ofertas_normalizadas")
                    df_feat = build_features(df_hist_imp, st.session_state["inventario"], perfil_f, cal_f, ofertas_n)
                    df_feat = cold_start_proxy(df_feat, st.session_state["inventario"])
                    model, metricas, rmse_cn = entrenar_modelo_ml(df_feat)
                    if model is not None:
                        guardar_modelo_farmacia(model, metricas, rmse_cn)
                        invalidar_cache_modelo()
                        st.success(f"\u2705 Modelo entrenado | RMSE: {metricas['rmse']} | R\u00b2: {metricas['r2']}")
                        fig_imp = grafico_importancia_features(metricas)
                        if fig_imp: st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.error(f"Error: {metricas.get('error','Desconocido')}")

    # --- Estado ---
    st.divider()
    st.markdown("#### Estado")
    c1, c2, c3, c4 = st.columns(4)
    inv_ok = "inventario" in st.session_state
    hist_ok = "historico" in st.session_state
    of_ok = "ofertas_normalizadas" in st.session_state
    ml_ok, _, _ = obtener_modelo_cacheado()
    with c1: st.markdown(f"{'\u2705' if inv_ok else '\u23f3'} **Inventario**")
    with c2: st.markdown(f"{'\u2705' if hist_ok else '\u23f3'} **Historico**")
    with c3: st.markdown(f"{'\u2705' if of_ok else '\u23f3'} **Ofertas**")
    with c4: st.markdown(f"{'\u2705' if ml_ok else '\u23f3'} **Modelo ML**")


# ===========================================================================
# MODULO 2: BUSINESS INTELLIGENCE
# ===========================================================================
def modulo_business_intelligence():
    if "inventario" not in st.session_state or "historico" not in st.session_state:
        st.warning("\u26a0\ufe0f Carga datos en Configuracion."); return
    df_inv = st.session_state["inventario"]
    df_hist = st.session_state["historico"]

    st.markdown("### \U0001f4ca Business Intelligence")

    # --- HS fijo a 2 meses ---
    meses_hs = HEALTH_SCORE_MESES_DEFAULT
    df_vm = obtener_ventas_media(meses_cobertura=max(1, int(meses_hs)))

    hs = calcular_health_score(df_inv, df_vm, meses_hs)
    df_z = calcular_stock_zombie(df_inv, df_hist)
    dz = df_z["Valor_Inmovilizado"].sum() if "Valor_Inmovilizado" in df_z.columns else 0
    df_rot = calcular_roturas(df_inv, df_vm)
    nr = len(df_rot)
    df_of = st.session_state.get("ofertas_normalizadas"); prot = cargar_productos_protegidos()
    df_prev = generar_pedido_cobertura(df_inv, df_vm, meses_hs, df_of, productos_protegidos=prot)
    ahorro = df_prev["Ahorro"].sum() if not df_prev.empty else 0
    coste = df_prev["Coste_Con_Dto"].sum() if not df_prev.empty else 0
    tend = df_vm["Factor_Tendencia"].mean()*100 if "Factor_Tendencia" in df_vm.columns else 0
    # Nuevos calculos
    df_rotacion = calcular_rotacion_stock(df_inv, df_vm)
    rotacion_media = df_rotacion["Rotacion"].mean() if not df_rotacion.empty else 0
    coste_op, df_coste_op = calcular_coste_oportunidad(df_inv, df_vm)
    df_uvi = calcular_stock_uvi(df_inv, df_hist)
    duvi = df_uvi["Valor_Inmovilizado"].sum() if not df_uvi.empty and "Valor_Inmovilizado" in df_uvi.columns else 0
    benchmark = calcular_benchmark_hs(hs)

    # === 1. KPIs principales ===
    st.markdown("---")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: render_kpi("Health Score", f"{hs}%")
    with k2: render_kpi("Zombie", format_eur(dz), f"{len(df_z)} prods", False)
    with k3: render_kpi("UVI", format_eur(duvi), f"{len(df_uvi)} prods", False)
    with k4: render_kpi("Roturas", str(nr), f"{format_eur(coste_op)}/mes", False)
    with k5: render_kpi("Rotacion", f"{rotacion_media:.2f}", "Media (>1=buen giro)")
    with k6: render_kpi("Tend. Ventas", f"{tend:+.1f}%", "vs año anterior")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("\U0001f4f8 Guardar Snapshot KPIs"):
        registrar_snapshot_kpi(hs, len(df_z), dz, nr, ahorro, coste)
        st.success("\u2705 Guardado.")

    # === 2. Gauge HS + Benchmark + Evolucion proyectada ===
    cg, ce = st.columns([1, 2])
    with cg:
        st.plotly_chart(grafico_gauge_health(hs), use_container_width=True, config={"displayModeBar":False})
        if benchmark.get("media_red") is not None:
            diff = hs - benchmark["media_red"]
            color = "green" if diff >= 0 else "red"
            st.markdown(f"""
            <div style="background:#F8F9FC;border-radius:10px;padding:10px;text-align:center;margin-top:-10px;">
                <span style="font-size:0.8rem;color:#64748B;">BENCHMARK RED ({benchmark['n_farmacias']} farmacias)</span><br>
                <span style="font-size:1.3rem;font-weight:700;">Media: {benchmark['media_red']}%</span>
                <span style="color:{color};font-weight:700;font-size:1.1rem;"> ({diff:+.1f}%)</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.caption("Ref: 2 meses | Benchmark: solo disponible con red activa")
    with ce:
        fig_cal = grafico_calendario_reposicion(df_inv, df_vm)
        if fig_cal:
            st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Sin datos de ventas para calcular calendario de reposicion.")

    # === 3. Dinero en Riesgo + Productos con peor rotacion ===
    st.markdown("---"); st.markdown("#### \U0001f4b0 Dinero en Riesgo")
    c_dr, c_top = st.columns([3, 2])
    with c_dr:
        fig_dr = grafico_dinero_en_riesgo_donut(dz, 0, coste_op)
        if fig_dr:
            st.plotly_chart(fig_dr, use_container_width=True, config={"displayModeBar":False})
        else:
            st.success("\u2705 Sin dinero en riesgo")
    with c_top:
        # Top productos con peor ratio rotacion (calculados con datos actuales)
        st.markdown("**\U0001f4c9 Productos con peor rotacion**")
        st.caption("Stock alto relativo a sus ventas")
        if not df_rotacion.empty:
            df_peor = df_rotacion[df_rotacion["Rotacion"] < 0.3].head(10)
            if not df_peor.empty:
                cols_show = [c for c in [COL_NOMBRE, COL_STOCK, "Venta_Media_Mensual", "Rotacion"] if c in df_peor.columns]
                st.dataframe(df_peor[cols_show], hide_index=True, use_container_width=True)
            else:
                st.success("\u2705 Todos los productos con buena rotacion")
        else:
            st.info("Carga datos para ver rotacion.")

    # === 4. Waterfall de Ahorro ===
    pedidos_hist = cargar_json_farmacia("historico_pedidos.json", default=[])
    if pedidos_hist:
        st.markdown("---"); st.markdown("#### \U0001f4b6 Ahorro Generado por Pedido")
        fig_wf = grafico_waterfall_ahorro(pedidos_hist)
        if fig_wf:
            st.plotly_chart(fig_wf, use_container_width=True, config={"displayModeBar":False})
        ahorro_total = sum(p.get("ahorro_ofertas", 0) for p in pedidos_hist)
        coste_total = sum(p.get("coste_total", 0) for p in pedidos_hist)
        c1, c2, c3 = st.columns(3)
        with c1: render_kpi("Ahorro Ofertas", format_eur(ahorro_total), "Dctos aplicados")
        with c2: render_kpi("Inversion Total", format_eur(coste_total))
        with c3: render_kpi("Pedidos", str(len(pedidos_hist)))

    # === 5. Evolucion HS (mantener) ===
    st.markdown("---"); st.markdown("#### \U0001f4c8 Evolucion Historica")
    df_hk = obtener_historico_kpi()
    if not df_hk.empty and len(df_hk) >= 2:
        fh, fzr = grafico_historico_kpi(df_hk)
        c1, c2 = st.columns(2)
        with c1:
            if fh: st.plotly_chart(fh, use_container_width=True, config={"displayModeBar":False})
        with c2:
            if fzr: st.plotly_chart(fzr, use_container_width=True, config={"displayModeBar":False})
        mejora = df_hk.iloc[-1]["health_score"] - df_hk.iloc[0]["health_score"]
        mejora_z = df_hk.iloc[0].get("n_zombies", 0) - df_hk.iloc[-1].get("n_zombies", 0)
        c1,c2,c3 = st.columns(3)
        with c1: render_kpi("Mejora Score", f"{mejora:+.1f}%", delta_positive=mejora>0)
        with c2: render_kpi("Zombies Eliminados", str(int(mejora_z)), delta_positive=mejora_z>0)
        with c3: render_kpi("Sesiones", str(len(df_hk)))
    else:
        st.info("Guarda al menos 2 snapshots para ver evolucion.")

    # === 6. ROI por Laboratorio ===
    st.markdown("---"); st.markdown("#### \U0001f4ca ROI por Laboratorio")
    df_roi = calcular_roi_laboratorios(df_inv, df_vm)
    if not df_roi.empty:
        fig_roi = grafico_roi_laboratorios(df_roi)
        if fig_roi:
            st.plotly_chart(fig_roi, use_container_width=True, config={"displayModeBar":False})
        with st.expander("Ver tabla completa de ROI"):
            df_roi_show = df_roi.copy()
            df_roi_show["Stock_EUR"] = df_roi_show["Stock_EUR"].apply(lambda x: f"{x:,.2f} \u20ac")
            df_roi_show["Venta_Anual_EUR"] = df_roi_show["Venta_Anual_EUR"].apply(lambda x: f"{x:,.2f} \u20ac")
            df_roi_show["ROI"] = df_roi_show["ROI"].apply(lambda x: f"{x:.2f}x")
            st.dataframe(df_roi_show, use_container_width=True, hide_index=True)

    # === 7. Rotacion de Stock + Coste de Oportunidad ===
    st.markdown("---"); st.markdown("#### \U0001f504 Rotacion de Stock")
    st.caption("Rotacion = Venta Mensual / Stock. Solo productos CON stock (stock=0 = rotura, se muestra en Roturas).")
    c_rot, c_co = st.columns(2)
    with c_rot:
        if not df_rotacion.empty:
            # Semaforo
            df_rot_show = df_rotacion.copy()
            def _semaforo_rot(r):
                if r >= 1.0: return "\U0001f7e2 Buena"
                if r >= 0.3: return "\U0001f7e1 Ajustar"
                return "\U0001f534 Sobrestock"
            df_rot_show["Estado"] = df_rot_show["Rotacion"].apply(_semaforo_rot)
            # Meses_Stock: reemplazar NaN por "Sin ventas"
            df_rot_show["Meses_Stock"] = df_rot_show["Meses_Stock"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "Sin ventas")
            n_sobre = len(df_rot_show[df_rot_show["Rotacion"] < 0.3])
            st.caption(f"Rotacion media: **{rotacion_media:.2f}** | \U0001f534 Sobrestock (<0.3): **{n_sobre}** prods")
            with st.expander("Ver ranking completo"):
                cols_show = [c for c in [COL_CN, COL_NOMBRE, COL_LAB, COL_STOCK, "Venta_Media_Mensual",
                             "Rotacion", "Meses_Stock", "Estado"] if c in df_rot_show.columns]
                st.dataframe(df_rot_show[cols_show].head(30), use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos de rotacion.")
    with c_co:
        st.markdown("**\U0001f4c9 Coste de Oportunidad de Roturas**")
        if coste_op > 0:
            st.metric("Venta perdida estimada", f"{coste_op:,.2f} \u20ac/mes")
            if not df_coste_op.empty:
                with st.expander(f"Ver {len(df_coste_op)} productos en rotura"):
                    st.dataframe(df_coste_op.head(20), use_container_width=True, hide_index=True)
        else:
            st.success("\u2705 Sin roturas — no hay venta perdida")

    # === 8. Zombie + UVI ===
    st.markdown("---"); st.markdown("#### \U0001f9df Stock Inmovilizado")
    c_z, c_u = st.columns(2)
    with c_z:
        st.markdown(f"**Zombie (12m sin ventas)** — {len(df_z)} prods | {format_eur(dz)}")
        if not df_z.empty:
            cs = [c for c in [COL_CN,COL_NOMBRE,COL_LAB,COL_STOCK,COL_PVL,"Valor_Inmovilizado"] if c in df_z.columns]
            st.dataframe(df_z[cs].sort_values("Valor_Inmovilizado",ascending=False).head(20), use_container_width=True, hide_index=True)
        else:
            st.success("\u2705 Sin stock zombie")
    with c_u:
        st.markdown(f"**UVI (6m sin ventas)** — {len(df_uvi)} prods | {format_eur(duvi)}")
        if not df_uvi.empty:
            cs = [c for c in [COL_CN,COL_NOMBRE,COL_LAB,COL_STOCK,COL_PVL,"Valor_Inmovilizado"] if c in df_uvi.columns]
            st.dataframe(df_uvi[cs].sort_values("Valor_Inmovilizado",ascending=False).head(20), use_container_width=True, hide_index=True)
        else:
            st.success("\u2705 Sin stock UVI")

    # === 9. Modelo ML (mantener) ===
    model, met, _ = obtener_modelo_cacheado()
    if model and met:
        st.markdown("---"); st.markdown("#### \U0001f9e0 Rendimiento del Modelo ML")
        c1,c2,c3 = st.columns(3)
        with c1: render_kpi("RMSE", str(met.get("rmse","?")))
        with c2: render_kpi("R\u00b2 Score", str(met.get("r2","?")))
        with c3: render_kpi("Features", str(len(met.get("features",[]))))
        fig_imp = grafico_importancia_features(met)
        if fig_imp: st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar":False})

    # === 10. Descargar Informe ===
    st.markdown("---")
    farmacia_nombre = st.session_state.get("farmacia_activa", "Farmacia").replace("_", " ").title()
    ahorro_acum_total = df_hk["ahorro_pedido"].sum() if not df_hk.empty else 0
    informe_bytes = generar_informe_pdf(
        farmacia_nombre, hs, len(df_z), dz, len(df_uvi), duvi,
        nr, coste_op, ahorro_acum_total, rotacion_media, benchmark)
    
    informe_filename = "informe_pharmaflow.pdf"
    
    # GUARDA FISICAMENTE COMO BACKUP 100% SEGURO
    with open(informe_filename, "wb") as f:
        f.write(informe_bytes)
        
    st.success(f"✅ Informe guardado físicamente en la carpeta como **{informe_filename}**.")
    st.info("💡 Si el botón de abajo te descarga un archivo sin extensión, simplemente renómbralo a '.pdf' o abre directamente el archivo guardado en tu carpeta.")
    
    # INTENTO DE DESCARGA NATIVA
    with open(informe_filename, "rb") as f:
        pdf_data = f.read()
        
    st.download_button(
        label="📄 Intentar Descargar Informe (PDF)",
        data=pdf_data, 
        file_name=informe_filename,
        mime="application/pdf", 
        width="stretch"
    )

# ===========================================================================
# MODULO 3: GENERADOR DE PEDIDOS (con meses_cobertura aqui)
# ===========================================================================
def modulo_generador_pedidos():
    if "inventario" not in st.session_state or "historico" not in st.session_state:
        st.warning("\u26a0\ufe0f Carga datos en Configuracion."); return

    df_inv = st.session_state["inventario"]
    df_ofertas = st.session_state.get("ofertas_normalizadas")
    protegidos = cargar_productos_protegidos()

    st.markdown("### \U0001f680 Generador de Pedidos Transfer")

    # --- 1. Laboratorio ---
    st.markdown("#### 1. Laboratorio")
    labs = ["-- Todos --"]
    if COL_LAB in df_inv.columns:
        labs += sorted(df_inv[COL_LAB].dropna().unique().tolist())
    lab_sel = st.selectbox("Laboratorio:", labs, key="lab_transfer")

    # --- 2. Parametros (meses cobertura AQUI) ---
    st.markdown("#### 2. Parametros")
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        modo = st.radio("Modo:", ["\U0001f4c5 Cobertura", "\U0001f4b0 Presupuesto"], horizontal=True, key="modo_ped")
    with cp2:
        meses = st.slider("Meses cobertura:", 1, 6, 2, key="meses_cob")
        presupuesto = 0
        if "Presupuesto" in modo:
            presupuesto = st.number_input("Presupuesto (\u20ac):", 100, 100000, 2000, step=100, key="ppto")
    with cp3:
        pedido_min = st.number_input("Pedido minimo (\u20ac):", 0, 10000, 100, step=25, key="ped_min")

    # --- 3. Motor de prediccion ---
    st.markdown("#### 3. Motor de Prediccion")
    model, met, rmse_cn = obtener_modelo_cacheado()
    opciones_motor = ["\U0001f4d0 Heuristico (Mirroring + Tendencia)"]
    if model and ML_AVAILABLE:
        r2_txt = met.get("r2", "?") if met else "?"
        opciones_motor.append(f"\U0001f9e0 ML Optimizado (R\u00b2={r2_txt})")
        opciones_motor.append(f"\U0001f500 Ensemble ML+Heuristico (R\u00b2={r2_txt})")
    motor = st.radio("Motor:", opciones_motor, horizontal=True, key="motor_pred")
    usar_ensemble = "Ensemble" in motor and model is not None
    usar_ml = not usar_ensemble and "ML" in motor and model is not None

    nivel_servicio = 95
    if usar_ml or usar_ensemble:
        nivel_servicio = st.radio("Nivel de Servicio:", [95, 99], horizontal=True, key="nivel_srv",
            format_func=lambda x: f"{x}% {'(Eficiencia)' if x == 95 else '(Seguridad)'}")

    # --- Generar ---
    st.markdown("---")
    if st.button("\u26a1 Generar Pedido", type="primary", use_container_width=True):
        with st.spinner("Calculando..."):
            df_f = df_inv.copy()
            if lab_sel != "-- Todos --" and COL_LAB in df_f.columns:
                df_f = df_f[df_f[COL_LAB] == lab_sel]

            if usar_ml or usar_ensemble:
                perfil_f = cargar_perfil_farmacia()
                cal_f = cargar_calendario_farmacia()
                df_hist_imp = imputar_stockouts(st.session_state["historico"], df_inv)
                df_feat = build_features(df_hist_imp, df_inv, perfil_f, cal_f, df_ofertas)
                df_feat = cold_start_proxy(df_feat, df_inv)
                ultimo_anio = df_feat["Anio"].max() if "Anio" in df_feat.columns else 2025
                ultimo_mes = df_feat.loc[df_feat["Anio"] == ultimo_anio, "Mes"].max() if "Mes" in df_feat.columns else 12
                df_futuro = df_feat[(df_feat["Anio"] == ultimo_anio) & (df_feat["Mes"] == ultimo_mes)].copy()
                if df_futuro.empty:
                    df_futuro = df_feat.groupby(COL_CN).last().reset_index()
                if lab_sel != "-- Todos --":
                    cns_lab = set(df_f[COL_CN])
                    df_futuro = df_futuro[df_futuro[COL_CN].isin(cns_lab)]

                if usar_ensemble:
                    df_vm_heur = obtener_ventas_media(meses_cobertura=meses)
                    ped = generar_pedido_ensemble(df_f, model, df_futuro, rmse_cn,
                                                  df_vm_heur, meses, nivel_servicio,
                                                  df_ofertas, protegidos)
                    st.session_state["motor_usado"] = "Ensemble"
                else:
                    ped = generar_pedido_ml(df_f, model, df_futuro, rmse_cn, meses,
                                            nivel_servicio, df_ofertas, protegidos)
                    st.session_state["motor_usado"] = "ML"
            else:
                df_vm = obtener_ventas_media(meses_cobertura=meses)
                if "Cobertura" in modo:
                    ped = generar_pedido_cobertura(df_f, df_vm, meses, df_ofertas,
                                                   productos_protegidos=protegidos)
                else:
                    ped = generar_pedido_presupuesto(df_f, df_vm, presupuesto, meses,
                                                      df_ofertas, protegidos)
                st.session_state["motor_usado"] = "Heuristico"

            st.session_state["pedido_generado"] = ped

    # --- Resultados ---
    df_ped = st.session_state.get("pedido_generado")
    motor_usado = st.session_state.get("motor_usado", "")
    if df_ped is not None and not df_ped.empty:
        if motor_usado:
            st.info(f"Motor utilizado: **{motor_usado}**")

        tc = df_ped["Coste_Con_Dto"].sum()
        ta = df_ped["Ahorro"].sum()
        tu = df_ped["Cantidad_A_Pedir"].sum()

        k1, k2, k3, k4 = st.columns(4)
        with k1: render_kpi("Total Pedido", format_eur(tc))
        with k2: render_kpi("Ahorro", format_eur(ta), f"{safe_div(ta, tc+ta)*100:.1f}%", True)
        with k3: render_kpi("Lineas", str(len(df_ped)))
        with k4: render_kpi("Unidades", f"{int(tu):,}".replace(",", "."))

        if "Presupuesto" in modo and presupuesto > 0:
            pct = min(100, safe_div(tc, presupuesto) * 100)
            bar_col = COLORS["success"] if pct < 80 else (COLORS["warning"] if pct < 95 else COLORS["danger"])
            st.markdown(f"""
            <div style="margin:1rem 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:0.85rem;font-weight:600;">Presupuesto</span>
                    <span style="font-size:0.85rem;font-weight:600;">{format_eur(tc)} / {format_eur(presupuesto)}</span>
                </div>
                <div class="budget-bar-container">
                    <div class="budget-bar-fill" style="width:{pct}%;background:{bar_col};">{pct:.0f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        df_ped_display = df_ped.copy()
        if COL_LAB in df_ped_display.columns:
            lab_list = sorted(df_ped_display[COL_LAB].dropna().unique())
        else:
            df_ped_display[COL_LAB] = "General"; lab_list = ["General"]

        for lab in lab_list:
            dl = df_ped_display[df_ped_display[COL_LAB] == lab]
            pl = dl["Coste_Con_Dto"].sum()
            al = dl["Ahorro"].sum()
            ups = dl["Upselling"].any() if "Upselling" in dl.columns else False
            cumple = pl >= pedido_min
            status = "\u2705" if cumple else "\u274c"
            ups_txt = " | \U0001f7e2 Upselling" if ups else ""

            with st.expander(f"\U0001f3ed **{lab}** | {format_eur(pl)} | Ahorro: {format_eur(al)} | {status}{ups_txt}",
                             expanded=(len(lab_list) == 1)):
                cols_d = [c for c in [COL_CN, COL_NOMBRE, COL_STOCK, "Venta_Media_Mensual",
                    "Safety_Stock", "Cantidad_A_Pedir", "Precio_Unitario", "Descuento_Aplicado",
                    "Coste_Con_Dto", "Ahorro", "Tier_Aplicado", "Upselling"] if c in dl.columns]
                dd = dl[cols_d].copy()
                if "Descuento_Aplicado" in dd.columns:
                    dd["Descuento_Aplicado"] = (dd["Descuento_Aplicado"] * 100).round(1).astype(str) + "%"
                if "Venta_Media_Mensual" in dd.columns:
                    dd["Venta_Media_Mensual"] = dd["Venta_Media_Mensual"].round(1)
                if "Safety_Stock" in dd.columns:
                    dd["Safety_Stock"] = dd["Safety_Stock"].round(0).astype(int)
                if "Upselling" in dd.columns:
                    dd["Upselling"] = dd["Upselling"].map({True: "\U0001f7e2 Si", False: ""})
                st.dataframe(dd, use_container_width=True, hide_index=True)

        st.markdown("---")
        lab_txt = lab_sel if lab_sel != "-- Todos --" else "Todos"
        col_dl, col_confirm = st.columns(2)
        with col_dl:
            st.download_button(
                f"\U0001f4e5 Descargar Pedido {lab_txt} (.xlsx)",
                data=exportar_pedido_excel(df_ped),
                file_name=f"PharmaFlow_{lab_txt}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary", use_container_width=True)
        with col_confirm:
            if st.button("\u2705 Confirmar Pedido", use_container_width=True, type="secondary"):
                modo_conf = "presupuesto" if "Presupuesto" in modo else "cobertura"
                df_vm_conf = obtener_ventas_media(meses_cobertura=meses)
                registrar_promociones_pedido(df_ped)
                # Registrar por laboratorio real (no "Todos")
                if lab_sel == "-- Todos --" and COL_LAB in df_ped.columns:
                    labs_en_pedido = df_ped[COL_LAB].dropna().unique()
                    for lab_real in labs_en_pedido:
                        df_ped_lab = df_ped[df_ped[COL_LAB] == lab_real]
                        registrar_pedido_confirmado(df_ped_lab, str(lab_real), modo_conf, meses, df_vm_conf)
                    reg_count = len(labs_en_pedido)
                    total_coste = round(float(df_ped["Coste_Con_Dto"].sum()), 2)
                    total_ahorro = round(float(df_ped["Ahorro"].sum()), 2)
                    # Calcular cobertura global (minimo de todos)
                    vm_map_conf = df_vm_conf.set_index(COL_CN)["Venta_Media_Mensual"].to_dict() if not df_vm_conf.empty else {}
                    dias_min = 999
                    cuello = ""
                    for _, row in df_ped.iterrows():
                        cn = row.get(COL_CN, "")
                        cant = int(row.get("Cantidad_A_Pedir", 0))
                        vmd = vm_map_conf.get(cn, 0) / 30.44
                        if vmd > 0:
                            d = cant / vmd
                            if d < dias_min:
                                dias_min = d
                                cuello = str(row.get(COL_NOMBRE, ""))[:50]
                    dias = round(dias_min) if dias_min < 999 else (meses or 2) * 30
                else:
                    lab_real = lab_sel if lab_sel != "-- Todos --" else "General"
                    reg = registrar_pedido_confirmado(df_ped, lab_real, modo_conf, meses, df_vm_conf)
                    reg_count = 1
                    total_coste = reg["coste_total"] if reg else 0
                    total_ahorro = reg["ahorro_ofertas"] if reg else 0
                    dias = reg["dias_cobertura_estimados"] if reg else (meses or 2) * 30
                    cuello = reg["producto_cuello_botella"] if reg else ""

                # Snapshot KPI automatico
                if "inventario" in st.session_state:
                    df_inv_c = st.session_state["inventario"]
                    hs = calcular_health_score(df_inv_c, df_vm_conf)
                    df_z = calcular_stock_zombie(df_inv_c, st.session_state.get("historico", pd.DataFrame()))
                    dz_val = df_z["Valor_Inmovilizado"].sum() if "Valor_Inmovilizado" in df_z.columns else 0
                    nr = len(calcular_roturas(df_inv_c, df_vm_conf))
                    registrar_snapshot_kpi(hs, len(df_z), dz_val, nr, total_ahorro, total_coste)

                prox = (date.today() + timedelta(days=max(0, dias - MARGEN_SEGURIDAD_DIAS))).strftime("%d/%m/%Y")
                st.session_state.pop("alertas_red_cache", None)  # Invalidar cache tras confirmar
                st.success(f"\u2705 Pedido confirmado ({reg_count} lab{'s' if reg_count > 1 else ''}) | "
                          f"Cobertura: ~{dias} dias | Proximo pedido sugerido: {prox}")
                if cuello:
                    st.caption(f"Cuello de botella: {cuello}")
                if pedido_min > 0 and total_coste < pedido_min:
                    st.warning(f"\u26a0\ufe0f Pedido ({format_eur(total_coste)}) por debajo del minimo ({format_eur(pedido_min)})")
                # Validacion de surtido post-pedido
                reglas_s = cargar_reglas_surtido()
                if reglas_s and "inventario" in st.session_state:
                    alertas_s = validar_surtido_pedido(st.session_state["inventario"], df_ped, reglas_s)
                    for alerta in alertas_s:
                        st.warning(alerta["mensaje"])

    elif df_ped is not None and df_ped.empty:
        st.info("\u2139\ufe0f Stock suficiente \u2014 no se necesitan compras.")

    # --- Panel de Surtido ---
    st.markdown("---")
    st.markdown("#### 🧪 Panel de Surtido Avanzado")
    st.caption("Diversificacion de laboratorios. Configura reglas con Dosis/Cant. en Configuracion > Reglas de Surtido.")
    if "inventario" in st.session_state:
        reglas_panel = cargar_reglas_surtido()
        df_surtido = generar_panel_surtido(st.session_state["inventario"], reglas_panel)
        if not df_surtido.empty:
            
            # --- Filtros Activos UI ---
            col_fT, col_fL = st.columns(2)
            with col_fT:
                tipos_disp = ["Todos"] + sorted([t.strip() for ts in df_surtido["Tipo"].unique() for t in ts.split(",") if t.strip()])
                tipos_disp = list(dict.fromkeys(tipos_disp)) # Remove duplicates
                filtro_tipo = st.selectbox("🗂️ Tipo de Producto:", options=tipos_disp, index=0)
            
            with col_fL:
                all_labs = list(set([l.strip() for row in df_surtido["Labs"].dropna() for l in row.split(",") if l.strip()]))
                all_labs.sort()
                filtro_labs = st.multiselect("🏭 Filtrar por Laboratorio/s (opc):", options=all_labs)

            df_surt_filtered = df_surtido.copy()
            if filtro_tipo != "Todos":
                df_surt_filtered = df_surt_filtered[df_surt_filtered["Tipo"].str.contains(filtro_tipo, na=False, case=False)]
            if filtro_labs:
                mask = df_surt_filtered["Labs"].apply(lambda labs_str: any(lab in labs_str for lab in filtro_labs) if pd.notna(labs_str) else False)
                df_surt_filtered = df_surt_filtered[mask]
                
            n_moleculas = len(df_surt_filtered)
            n_mono = len(df_surt_filtered[df_surt_filtered["N_Labs"] == 1])
            col1, col2, col3 = st.columns(3)
            with col1: render_kpi("Productos Base (Filtro)", f"{n_moleculas:,}")
            with col2: render_kpi("Monolab", f"{n_mono:,}")
            with col3:
                if reglas_panel:
                    n_incumple = len(df_surt_filtered[df_surt_filtered["Cumple"] == "❌"])
                    render_kpi(f"{'⚠️' if n_incumple > 0 else '✅'} Alertas Surtido Min.", f"{n_incumple}")
                else: render_kpi("ℹ️ Reglas Activas", "0")

            with st.expander(f"Ver tabla de surtido ({n_moleculas} registros)"):
                st.dataframe(df_surt_filtered.rename(columns={"Presentacion/Molecula": "Regla / Molécula"}), use_container_width=True, hide_index=True)
        else:
            st.info("Carga inventario para ver el panel de surtido.")
    else:
        st.info("Carga inventario en Configuracion.")


# ===========================================================================
# MODULO 4: AUDITORIA
# ===========================================================================
def modulo_auditoria():
    if "inventario" not in st.session_state or "historico" not in st.session_state:
        st.warning("\u26a0\ufe0f Carga datos en Configuracion."); return

    df_inv = st.session_state["inventario"]
    df_hist = st.session_state["historico"]
    df_vm = obtener_ventas_media(meses_cobertura=int(HEALTH_SCORE_MESES_DEFAULT))

    st.markdown("### \U0001f6e0\ufe0f Auditoría Integral de Stock")
    
    # Pre-calculo de variables Core
    hs = calcular_health_score(df_inv, df_vm)
    df_z = calcular_stock_zombie(df_inv, df_hist)
    dz = df_z["Valor_Inmovilizado"].sum() if not df_z.empty and "Valor_Inmovilizado" in df_z.columns else 0
    df_u = calcular_stock_uvi(df_inv, df_hist)
    du = df_u["Valor_Inmovilizado"].sum() if not df_u.empty and "Valor_Inmovilizado" in df_u.columns else 0
    df_rot = calcular_roturas(df_inv, df_vm)
    
    c_op, _ = calcular_coste_oportunidad(df_inv, df_vm)
    
    # Nuevas variables
    fill_rate, total_con_demanda, n_roturas_hab = calcular_indice_servicio(df_inv, df_hist)
    val_cad, prods_cad, cad_real = calcular_riesgo_caducidad(df_inv)
    
    
    # --- RESUMEN EJECUTIVO (Scorecard) ---
    st.markdown("#### \U0001f4cb Resumen Ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(grafico_gauge_health(hs), use_container_width=True, config={"displayModeBar":False}, key="gauge_auditoria")
    with c2:
        render_kpi("Índice Servicio (Fill Rate)", f"{fill_rate:.1f}%", f"{n_roturas_hab} roturas / {total_con_demanda} actvs", fill_rate>90)
        st.markdown("<br>", unsafe_allow_html=True)
        render_kpi("Caducidades (<6m)", format_eur(val_cad), "Real" if cad_real else "Estimado (Mock)", cad_real)
    with c3:
        # Calcular sobrestock para meter al donut
        df_m = df_inv.merge(df_vm, on=COL_CN, how="left")
        df_m["Venta_Media_Mensual"] = df_m["Venta_Media_Mensual"].fillna(0)
        df_m["Stock_Ideal"] = df_m["Venta_Media_Mensual"] * HEALTH_SCORE_MESES_DEFAULT
        df_m["Exceso"] = (df_m[COL_STOCK] - df_m["Stock_Ideal"]).clip(lower=0)
        val_sob = (df_m["Exceso"] * df_m.get(COL_PVL, pd.Series(0))).fillna(0).sum()
        
        fig_donut = grafico_dinero_en_riesgo_donut(dz, val_sob, c_op)
        if fig_donut:
            fig_donut.update_layout(height=180, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar":False}, key="donut_auditoria")
    with c4:
        # Comparativa temporal rapida
        hist_aud = obtener_historico_auditorias()
        if len(hist_aud) >= 2:
            prev = hist_aud[-2]; curr = hist_aud[-1]
            dz_kpi = curr["n_zombies"] - prev["n_zombies"]
            render_kpi("Comparativa Zombies", f"{curr['n_zombies']}", f"{dz_kpi:+d} vs anterior", dz_kpi <= 0)
        else:
            st.info("Guarda snapshots para ver la evolución vs auditoría previa.")
            
    st.markdown("---")
    
    # --- PESTAÑAS DE ANÁLISIS PROFUNDO ---
    ta1, ta2, ta3, ta4 = st.tabs([
        "📊 Distribución y Eficiencia", 
        "⚠️ Riesgo Operacional", 
        "💸 Flujo de Caja y Estacionalidad",
        "⚖️ Conciliación Físico-Lógico"
    ])
    
    with ta1:
        st.markdown("#### \U0001f4ca Distribución del Surtido e Inversión")
        c_l, c_r = st.columns([1.5, 1])
        with c_l:
            st.markdown("**Análisis de Pareto (Regla 80/20)**")
            fig_tail = grafico_long_tail(df_inv, df_vm)
            if fig_tail: st.plotly_chart(fig_tail, use_container_width=True, config={"displayModeBar":False}, key="tail_auditoria")
            
        with c_r:
            st.markdown("**Inversión por Laboratorio**")
            tipo_filtro = st.radio("Filtro:", ["Todos", "Medicamento", "Parafarmacia"], horizontal=True)
            fig_labs = grafico_distribucion_laboratorios(df_inv, tipo_filtro)
            if fig_labs: st.plotly_chart(fig_labs, use_container_width=True, config={"displayModeBar":False}, key="labs_auditoria")
            
        st.markdown("#### Análisis ABC (Valor de Inmovilizado)")
        df_abc = calcular_analisis_abc(df_inv, df_vm)
        if not df_abc.empty:
            df_abc_show = df_abc[[COL_NOMBRE, COL_LAB, COL_STOCK, "Valor_Stock", "Clasificacion_ABC", "Venta_Media_Mensual"]].head(25)
            st.dataframe(df_abc_show, use_container_width=True, hide_index=True)

    with ta2:
        st.markdown("#### \u26a0\ufe0f Riesgo Operacional (Inmovilizado)")
        c_z, c_u = st.columns(2)
        with c_z:
            st.markdown(f"**\U0001f9df Stock Zombie (12m sin ventas)** — {format_eur(dz)}")
            if not df_z.empty:
                st.dataframe(df_z[[COL_NOMBRE, COL_LAB, COL_STOCK, "Valor_Inmovilizado"]].sort_values("Valor_Inmovilizado", ascending=False).head(15), use_container_width=True, hide_index=True)
            else: st.success("Sin zombies")
        with c_u:
            st.markdown(f"**\U0001fa79 Stock UVI (6m sin ventas)** — {format_eur(du)}")
            if not df_u.empty:
                st.dataframe(df_u[[COL_NOMBRE, COL_LAB, COL_STOCK, "Valor_Inmovilizado"]].sort_values("Valor_Inmovilizado", ascending=False).head(15), use_container_width=True, hide_index=True)
            else: st.success("Sin UVI")
            
        st.markdown("---")
        c_hc, c_cr = st.columns([1.5, 1])
        with c_hc:
            fig_heat = grafico_heatmap_cobertura(df_inv, df_vm)
            if fig_heat: st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar":False}, key="heat_auditoria")
        with c_cr:
            df_exceso = df_m[df_m["Exceso"] > 0]
            fig_conc = grafico_concentracion_riesgo(df_z, df_u, df_exceso)
            if fig_conc: st.plotly_chart(fig_conc, use_container_width=True, config={"displayModeBar":False}, key="conc_auditoria")
            
    with ta3:
        st.markdown("#### \U0001f4b8 Flujo de Caja y Estacionalidad")
        c_fc, c_est = st.columns(2)
        
        with c_fc:
            st.markdown("**Ciclo de Conversión de Efectivo (CCC)**")
            st.caption("¿La farmacia se financia con proveedores o con su bolsillo?")
            
            dias_pago_manual = st.slider("Días promedio de pago a proveedores (Manual)", 0, 150, 60, 5)
            
            uploaded_facturas = st.file_uploader("Opcional: Sube CSV Facturas/Mayor Proveedores para cálculo exacto", type=["csv"], key="up_facts")
            df_facts = None
            if uploaded_facturas:
                df_facts = pd.read_csv(uploaded_facturas, encoding="utf-8")
                st.success("Facturas cargadas. Se usará media ponderada.")
                
            d_pago, d_inv, ccc, nec_fm, flotador = calcular_flujo_caja_inventario(df_inv, df_vm, dias_pago_manual, df_facts)
            
            if d_inv > 0:
                st.markdown(f"- **Días que el stock tarda en venderse:** {d_inv:.0f} días")
                st.markdown(f"- **Días promedio de pago:** {d_pago:.0f} días")
                if ccc > 0:
                    st.error(f"**Agujero Financiero:** El comprador necesita fondear {format_eur(nec_fm)} (Caja retenida {ccc:.0f} días)")
                else:
                    st.success(f"**Flotador Positivo:** Proveedores financian {format_eur(flotador)} al comprador (Caja adelantada {abs(ccc):.0f} días)")
            else:
                st.info("Sube datos con PVL para calcular.")
                
            st.markdown("---")
            st.markdown("**Matriz Rentabilidad (GMROI)**")
            df_gmroi, tiene_pvp = calcular_matriz_rentabilidad_gmroi(df_inv, df_vm)
            if tiene_pvp and not df_gmroi.empty:
                res = df_gmroi.groupby("Cuadrante")[COL_CN].count().reset_index()
                st.dataframe(res, use_container_width=True, hide_index=True)
            else:
                st.info("Se requiere columna PVP para calcular matriz real.")
                
        with c_est:
            st.markdown("**Dependencia Estacional (Riesgo de Liquidez)**")
            df_est = calcular_dependencia_estacional(df_hist)
            if not df_est.empty:
                fig_est = grafico_estacionalidad_liquidez(df_est)
                if fig_est: st.plotly_chart(fig_est, use_container_width=True, config={"displayModeBar":False}, key="est_auditoria")
                
                # Alerta si un trimestre acumula > 40% ventas
                max_3m = df_est["Pct_Ventas"].rolling(window=3, min_periods=1).sum().max()
                if max_3m > 40:
                    st.warning(f"\u26a0\ufe0f **Riesgo Estacional Alto:** El {max_3m:.1f}% de las ventas ocurre en solo 3 meses seguidos.")
                else:
                    st.success(f"\u2705 **Riesgo Estacional Bajo:** Ingresos estables (pico trimestral {max_3m:.1f}%).")
            else:
                st.info("Sin histórico suficiente para calcular estacionalidad.")
                
    with ta4:
        st.markdown("#### \u2696\ufe0f Conciliación Físico vs Lógico")
        st.caption("Cruza el stock teórico del ordenador con un archivo CSV exportado desde un lector de códigos de barras.")
        
        uploaded_fisico = st.file_uploader("Sube CSV Inventario Físico (Columnas: Codigo_Nacional, Cantidad)", type=["csv"], key="up_fisico")
        
        if uploaded_fisico:
            try:
                df_fisico = pd.read_csv(uploaded_fisico, sep=None, engine="python", encoding="utf-8")
                df_conc = calcular_conciliacion_fisico_logico(df_inv, df_fisico)
                
                if df_conc is not None and not df_conc.empty:
                    faltantes_eur = abs(df_conc[df_conc["Descuadre_Uds"] < 0]["Descuadre_Eur"].sum())
                    sobrantes_eur = df_conc[df_conc["Descuadre_Uds"] > 0]["Descuadre_Eur"].sum()
                    
                    c_f1, c_f2 = st.columns(2)
                    with c_f1: render_kpi("Faltantes (Pérdida)", format_eur(faltantes_eur), delta_positive=False)
                    with c_f2: render_kpi("Sobrantes (No registrados)", format_eur(sobrantes_eur), delta_positive=True)
                    
                    st.markdown("**Lista de Descuadres:**")
                    df_muest = df_conc[df_conc["Descuadre_Uds"] != 0].copy()
                    df_muest["Stock_Teorico"] = df_muest[COL_STOCK]
                    st.dataframe(df_muest[[COL_NOMBRE, "Stock_Teorico", "Stock_Fisico", "Descuadre_Uds", "Descuadre_Eur"]].sort_values("Descuadre_Eur"), use_container_width=True, hide_index=True)
                else:
                    st.error("No se pudieron detectar las columnas correctas en el CSV subido.")
            except Exception as e:
                st.error(f"Error procesando CSV: {str(e)}")
        else:
            st.info("Sube un CSV de conteo físico para visualizar el descuadre y detectar roturas o robos invisibles al software.")


# ===========================================================================
# MODULO 5: TORRE DE CONTROL (RED)
# ===========================================================================
def modulo_torre_control():
    st.markdown("### \U0001f3d7\ufe0f Torre de Control — Red de Farmacias")

    # --- 1. Gestion de Red ---
    st.markdown("#### \U0001f310 Red de Farmacias")
    red = cargar_red_config()
    farmacias_disponibles = obtener_farmacias_disponibles()
    farmacias_activas = red.get("farmacias_activas", [])

    c1, c2 = st.columns([2, 1])
    with c1:
        nuevas_activas = st.multiselect(
            "Farmacias en la red:",
            farmacias_disponibles,
            default=[f for f in farmacias_activas if f in farmacias_disponibles],
            key="red_farmacias")
        if st.button("\U0001f4be Guardar Red"):
            red["farmacias_activas"] = nuevas_activas
            guardar_red_config(red)
            st.session_state.pop("alertas_red_cache", None)  # Invalidar cache
            st.success(f"\u2705 Red actualizada: {len(nuevas_activas)} farmacias.")
    with c2:
        hist_conj = cargar_historico_compras_conjuntas()
        ahorro_total_red = sum(c.get("ahorro_total", 0) for c in hist_conj)
        render_kpi("Farmacias Red", str(len(farmacias_activas)))
        render_kpi("Compras Conjuntas", str(len(hist_conj)))
        render_kpi("Ahorro Red Total", format_eur(ahorro_total_red))

    if len(farmacias_activas) < 2:
        st.info("\u2139\ufe0f Necesitas al menos 2 farmacias en la red. Crea farmacias y anadelas arriba.")
        return

    # --- 2. Notificaciones globales (todas las alertas de la red) ---
    alertas = st.session_state.get("alertas_red_cache", [])
    if alertas:
        st.markdown("---")
        st.markdown(f"#### \U0001f514 Oportunidades Detectadas ({len(alertas)})")
        for i, al in enumerate(alertas):
            tipo_icon = "\U0001f7e2" if al["tipo"] == "natural" else "\U0001f7e1"
            farms_txt = " + ".join(f.replace("_", " ").title() for f in al["farmacias"])
            fecha_txt = al["fecha_sugerida"].strftime("%d/%m/%Y") if isinstance(al["fecha_sugerida"], date) else str(al["fecha_sugerida"])
            ajuste_txt = f" | \u23f3 {al['ajuste']}" if al.get("ajuste") else ""
            st.markdown(
                f"{tipo_icon} **{al['laboratorio']}**: {farms_txt} — "
                f"Semana del {fecha_txt} ({al['dias_disponibles']} dias ventana){ajuste_txt}")

    # --- 3. Calendario de Pedidos (Gantt) ---
    st.markdown("---")
    st.markdown("#### \U0001f4c5 Calendario de Pedidos")

    # Recoger labs disponibles
    labs_red = set()
    for farm in farmacias_activas:
        for p in cargar_pedidos_confirmados(farm):
            if p.get("laboratorio"):
                labs_red.add(p["laboratorio"])

    if not labs_red:
        st.info("\u2139\ufe0f No hay pedidos confirmados en la red. Confirma pedidos en la pestana 'Pedidos Transfer'.")
        return

    lab_filtro = st.selectbox("Filtrar por Laboratorio:", sorted(labs_red), key="tc_lab")

    ventanas = obtener_ventanas_red(lab_filtro)
    if not ventanas:
        st.info(f"No hay ventanas de pedido para {lab_filtro}.")
        return

    # Construir Gantt
    hoy = date.today()
    gantt_data = []
    for v in ventanas:
        farm_label = v["farmacia"].replace("_", " ").title()
        dias_opt = v["dias_hasta_optima"]
        color = "green" if dias_opt > 14 else ("gold" if dias_opt > 0 else "red")
        gantt_data.append({
            "Farmacia": farm_label,
            "Inicio": v["fecha_optima"],
            "Fin": v["fecha_limite"],
            "Color": color,
            "Dias": v["dias_hasta_limite"],
        })

    # Calcular oportunidades una sola vez para Gantt + simulacion
    oportunidades_calc = optimizar_timing_red(ventanas)

    if gantt_data:
        df_gantt = pd.DataFrame(gantt_data)
        fig_gantt = go.Figure()

        color_map = {"green": COLORS["success"], "gold": COLORS["warning"], "red": COLORS["danger"]}

        for _, row in df_gantt.iterrows():
            fig_gantt.add_trace(go.Bar(
                x=[(row["Fin"] - row["Inicio"]).days],
                y=[row["Farmacia"]],
                base=[(row["Inicio"] - hoy).days],
                orientation="h",
                marker_color=color_map.get(row["Color"], COLORS["primary"]),
                name=row["Farmacia"],
                showlegend=False,
                hovertemplate=f"{row['Farmacia']}<br>Optima: {row['Inicio']}<br>Limite: {row['Fin']}<extra></extra>",
            ))

        # Linea de hoy
        fig_gantt.add_vline(x=0, line_dash="dash", line_color=COLORS["text"],
                           annotation_text="Hoy", annotation_position="top")

        # Bandas doradas para solapamientos
        for op in oportunidades_calc:
            if op["dias_disponibles"] > 0:
                x0 = (op["fecha_sugerida"] - hoy).days
                x1 = (op["fin_ventana"] - hoy).days
                fig_gantt.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor="gold", opacity=0.15,
                    line_width=2, line_color="gold", line_dash="dot",
                    annotation_text="\U0001f91d",
                    annotation_position="top left",
                )

        fig_gantt.update_layout(
            title=f"Ventanas de Pedido — {lab_filtro}",
            xaxis_title="Dias desde hoy",
            height=max(200, len(gantt_data) * 60 + 100),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
            yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor="#E2E8F0"),
            barmode="overlay",
            margin=dict(l=150, r=30, t=50, b=40),
        )
        st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": False})

    # --- 4. Oportunidades de Compra Conjunta ---
    st.markdown("---")
    st.markdown("#### \U0001f91d Simulacion de Compra Conjunta")

    oportunidades_validas = [o for o in oportunidades_calc if o["dias_disponibles"] > 0]

    if not oportunidades_validas:
        st.info("No hay oportunidades de compra conjunta activas para este laboratorio.")
    else:
        for idx, op in enumerate(oportunidades_validas):
            farms = op["farmacias"]
            farms_txt = " + ".join(f.replace("_", " ").title() for f in farms)
            tipo_badge = "\U0001f7e2 Natural" if op["tipo"] == "natural" else "\U0001f7e1 Optimizado"

            with st.expander(f"\U0001f4e6 {farms_txt} | {tipo_badge} | Ventana: {op['dias_disponibles']} dias", expanded=(idx == 0)):
                if op.get("ajuste_necesario"):
                    st.info(f"\U0001f4a1 {op['ajuste_necesario']} (sin riesgo de rotura)")

                st.markdown(f"**Fecha sugerida:** {op['fecha_sugerida'].strftime('%d/%m/%Y') if isinstance(op['fecha_sugerida'], date) else op['fecha_sugerida']}")

                # Cargar pedidos individuales de cada farmacia
                pedidos_farms = []
                for farm in farms:
                    peds = cargar_pedidos_confirmados(farm)
                    peds_lab = [p for p in peds if p.get("laboratorio", "").lower() == lab_filtro.lower()]
                    if peds_lab:
                        ultimo = peds_lab[-1]
                        df_p = pd.DataFrame(ultimo.get("productos", []))
                        if not df_p.empty and "cn" in df_p.columns:
                            df_p = df_p.rename(columns={"cn": COL_CN, "nombre": COL_NOMBRE,
                                "cantidad": "Cantidad_A_Pedir", "molecula": COL_MOLECULA})
                            # Usar PVL original (sin descuento) para que la simulacion recalcule tiers limpiamente
                            df_p["Precio_Unitario"] = pd.to_numeric(df_p.get("pvl", pd.Series(0)), errors="coerce").fillna(0)
                            df_p["Descuento_Aplicado"] = 0.0
                            df_p["Coste_Sin_Dto"] = df_p["Cantidad_A_Pedir"] * df_p["Precio_Unitario"]
                            # Coste individual = con descuento original guardado
                            dto_original = pd.to_numeric(df_p.get("descuento", pd.Series(0)), errors="coerce").fillna(0)
                            df_p["Coste_Con_Dto"] = df_p["Coste_Sin_Dto"] * (1 - dto_original)
                            df_p["Ahorro"] = df_p["Coste_Sin_Dto"] - df_p["Coste_Con_Dto"]
                            pedidos_farms.append((farm, df_p))

                if len(pedidos_farms) >= 2:
                    df_ofertas = st.session_state.get("ofertas_normalizadas")
                    sim = simular_pedido_conjunto(pedidos_farms, df_ofertas)

                    if sim:
                        st.markdown("**Comparativa:**")
                        df_comp = pd.DataFrame(sim["resumen_farmacias"])
                        df_comp.columns = ["Farmacia", "Coste Individual", "Coste Conjunto", "AHORRO", "Unidades", "% Volumen"]
                        df_comp["Farmacia"] = df_comp["Farmacia"].str.replace("_", " ").str.title()

                        # Highlight ahorro
                        st.dataframe(df_comp, use_container_width=True, hide_index=True)

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            render_kpi("Ahorro Total", format_eur(sim["ahorro_total"]),
                                      delta_positive=True)
                        with c2:
                            coste_ind_total = sum(r["coste_individual"] for r in sim["resumen_farmacias"])
                            pct = safe_div(sim["ahorro_total"], coste_ind_total) * 100
                            render_kpi("% Ahorro", f"{pct:.1f}%", delta_positive=True)
                        with c3:
                            render_kpi("Unidades Conjunto",
                                      str(int(sim["df_conjunto"]["Cantidad_A_Pedir"].sum())))

                        # Barra de progreso al siguiente tier
                        ti = sim.get("tier_info")
                        if ti:
                            st.markdown("---")
                            st.markdown(f"**\U0001f3af Siguiente Tier:** {ti['producto']}")
                            actual = 0
                            if COL_NOMBRE in sim["df_conjunto"].columns:
                                mask_prod = sim["df_conjunto"][COL_NOMBRE] == ti["producto"]
                                actual = int(sim["df_conjunto"].loc[mask_prod, "Cantidad_A_Pedir"].sum())
                            objetivo = ti["siguiente_tier_min"]
                            pct_tier = min(100, safe_div(actual, objetivo) * 100)
                            st.markdown(
                                f"Actual: {actual} uds → Objetivo: {objetivo} uds ({ti['siguiente_tier_dto']}) — "
                                f"Faltan **{ti['faltan_uds']} uds**")
                            st.progress(pct_tier / 100)

                        # Botones
                        col_exec, col_lost = st.columns(2)
                        with col_exec:
                            if st.button(f"\u2705 Marcar Ejecutada", key=f"exec_{idx}"):
                                registrar_compra_conjunta(
                                    farms, lab_filtro, sim["ahorro_total"], sim["resumen_farmacias"])
                                st.success("\u2705 Compra conjunta registrada.")
                                st.rerun()
                        with col_lost:
                            if st.button(f"\u274c No Ejecutada", key=f"lost_{idx}"):
                                for r in sim["resumen_farmacias"]:
                                    registrar_ahorro_perdido(
                                        r["farmacia"], r["ahorro"], lab_filtro,
                                        f"Compra conjunta con {farms_txt} no realizada")
                                st.warning("Registrado como ahorro perdido.")
                                st.rerun()
                    else:
                        st.caption("No se pudo simular (faltan datos de ofertas o pedidos).")
                else:
                    st.caption("Confirma pedidos de ambas farmacias para ver la simulacion.")

    # --- 5. Coste de No Actuar ---
    st.markdown("---")
    st.markdown("#### \U0001f4b8 Coste de No Actuar")
    hay_datos_perdidos = False
    for farm in farmacias_activas:
        data = cargar_ahorro_perdido(farm)
        if data["total_perdido"] > 0:
            hay_datos_perdidos = True
            farm_label = farm.replace("_", " ").title()
            st.markdown(f"**{farm_label}:** {format_eur(data['total_perdido'])} en ahorro perdido "
                       f"({len(data['oportunidades'])} oportunidades no aprovechadas)")
    if not hay_datos_perdidos:
        st.success("\u2705 Todas las oportunidades han sido aprovechadas (o no hay datos aun).")

    # --- 6. Historial ---
    st.markdown("---")
    st.markdown("#### \U0001f4dc Historial de Compras Conjuntas")
    hist = cargar_historico_compras_conjuntas()
    if hist:
        rows = []
        for h in hist:
            farms_txt = ", ".join(f.replace("_", " ").title() for f in h.get("farmacias", []))
            rows.append({
                "Fecha": h["fecha"],
                "Laboratorio": h["laboratorio"],
                "Farmacias": farms_txt,
                "Ahorro Total": format_eur(h["ahorro_total"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No hay compras conjuntas registradas aun.")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    st.set_page_config(page_title=f"{APP_NAME} \u2014 Gestion de Compras",
        page_icon=APP_ICON, layout="wide", initial_sidebar_state="collapsed")
    inject_custom_css()

    if "farmacia_activa" not in st.session_state:
        render_header()
        modulo_selector_farmacia()
        return

    farmacia = st.session_state["farmacia_activa"]
    # Auto-cargar datos desde disco si no estan en session_state
    datos_cargados = _autocargar_datos_farmacia()
    if datos_cargados:
        st.toast(f"\U0001f4c2 Datos restaurados: {', '.join(datos_cargados)}", icon="\u2705")
    # Notificaciones (campanita) — cacheado en session_state para evitar lecturas redundantes
    if "alertas_red_cache" not in st.session_state:
        st.session_state["alertas_red_cache"] = generar_notificaciones()
    alertas = st.session_state["alertas_red_cache"]
    n_alertas = len(alertas)
    bell = f" \U0001f514 {n_alertas}" if n_alertas > 0 else ""
    render_header(farmacia_nombre=farmacia.replace("_", " ").title() + bell)

    with st.sidebar:
        st.markdown(f"**Farmacia:** {farmacia.replace('_', ' ').title()}")
        if n_alertas > 0:
            st.markdown(f"\U0001f514 **{n_alertas} oportunidades** de compra conjunta")
            for al in alertas[:3]:
                farms_txt = " + ".join(f.replace("_", " ").title() for f in al["farmacias"])
                st.caption(f"• {al['laboratorio']}: {farms_txt}")
        if st.button("\U0001f504 Cambiar Farmacia"):
            for k in list(st.session_state.keys()):
                if k != "farmacia_activa":
                    st.session_state.pop(k, None)
            st.session_state.pop("farmacia_activa", None)
            st.rerun()

    t1, t2, t3, t4, t5 = st.tabs([
        "\u2699\ufe0f Configuracion",
        "\U0001f4ca Business Intelligence",
        "\U0001f680 Pedidos Transfer",
        "\U0001f6e0\ufe0f Auditoria",
        "\U0001f3d7\ufe0f Torre de Control",
    ])
    with t1: modulo_configuracion()
    with t2: modulo_business_intelligence()
    with t3: modulo_generador_pedidos()
    with t4: modulo_auditoria()
    with t5: modulo_torre_control()

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:{COLORS['muted']};font-size:0.8rem;'>"
        f"{APP_ICON} {APP_NAME} v{VERSION} \u00b7 {farmacia.replace('_', ' ').title()} \u00b7 "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
