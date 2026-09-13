import json
import os
import streamlit as st
import pandas as pd
from config.settings import BASE_DIR, COL_CN
from utils.helpers import normalizar_cn

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
    # Escritura atomica: si el proceso muere a mitad, el archivo original
    # queda intacto en vez de corrompido a medio escribir.
    tmp = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, filepath)

def guardar_dataframe_farmacia(nombre_archivo, df):
    """Guarda un DataFrame como Parquet en la carpeta de la farmacia activa."""
    ruta = ruta_farmacia_activa()
    if ruta is None or df is None or df.empty:
        return
    filepath = ruta / nombre_archivo
    try:
        tmp = filepath.with_suffix(filepath.suffix + ".tmp")
        df.to_parquet(tmp, index=False, engine="pyarrow")
        os.replace(tmp, filepath)
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
            if COL_CN in df.columns:
                df[COL_CN] = normalizar_cn(df[COL_CN])
            st.session_state["inventario"] = df
            cargados.append(f"Inventario ({len(df)} productos)")
    if "historico" not in st.session_state:
        df = cargar_dataframe_farmacia("historico.parquet")
        if df is not None and not df.empty:
            if COL_CN in df.columns:
                df[COL_CN] = normalizar_cn(df[COL_CN])
            st.session_state["historico"] = df
            cargados.append(f"Historico ({len(df)} registros)")
    if "ofertas_normalizadas" not in st.session_state:
        df = cargar_dataframe_farmacia("ofertas_normalizadas.parquet")
        if df is not None and not df.empty:
            st.session_state["ofertas_normalizadas"] = df
            cargados.append(f"Ofertas ({len(df)} items)")
    return cargados
