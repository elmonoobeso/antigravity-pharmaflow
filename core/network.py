import json
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from config.settings import BASE_DIR, COL_CN, COL_LAB, COL_NOMBRE, COL_PVL, COL_MOLECULA
from data.io import ruta_farmacia_activa
from core.business import aplicar_ofertas_indexado, construir_indice_ofertas, buscar_oferta_por_indice

MARGEN_SEGURIDAD_DIAS = 14

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
