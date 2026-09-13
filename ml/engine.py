import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import pandas as pd
import numpy as np
from datetime import datetime
from calendar import monthrange
from data.io import cargar_json_farmacia, guardar_json_farmacia, ruta_farmacia_activa
from config.settings import (
    COL_CN, COL_VENTAS, COL_FECHA, COL_MOLECULA, COL_STOCK, Z_SCORES
)
from core.business import (
    calcular_horas_mes, calcular_dias_abiertos_mes, obtener_cns_con_promo_historica,
    generar_pedido_cobertura
)
from utils.helpers import safe_div
import streamlit as st

try:
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

TEMP_CLIMATICA = {
    "mediterraneo": [10.5, 11.2, 13.5, 15.8, 19.5, 24.0, 27.5, 27.2, 23.8, 18.5, 13.5, 10.8],
    "continental":  [5.5,  7.0,  10.5, 13.0, 17.0, 22.5, 26.0, 25.5, 20.5, 14.5, 9.0,  6.0],
    "atlantico":    [9.0,  9.5,  11.5, 12.5, 15.0, 18.0, 20.5, 20.8, 19.0, 15.5, 12.0, 9.8],
}

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

def _pct_zona_cobro(anio, mes):
    dias = monthrange(int(anio), int(mes))[1]
    dias_cobro = max(0, dias - 24)
    return round(dias_cobro / dias, 3)

def _obtener_mapa_atc(df_inventario):
    atc_tabla = st.session_state.get("tabla_maestra_atc")
    if atc_tabla is not None and not atc_tabla.empty:
        cn_col = [c for c in atc_tabla.columns if "codigo" in c.lower() or "cn" in c.lower() or "nacional" in c.lower()]
        atc_col = [c for c in atc_tabla.columns if "atc" in c.lower() or "grupo" in c.lower()]
        if cn_col and atc_col:
            mapa = atc_tabla.set_index(cn_col[0])[atc_col[0]].to_dict()
            return {str(k): str(v)[:5] for k, v in mapa.items() if pd.notna(v)}
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

def _obtener_temperatura(anio, mes, temp_data, zona):
    if temp_data is not None and not temp_data.empty:
        match = temp_data[(temp_data["Anio"] == anio) & (temp_data["Mes"] == mes)]
        if not match.empty:
            return round(float(match.iloc[0].get("Temp_Media", 0)), 1)
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

def build_features(df_ventas, df_inventario, perfil, calendario, df_ofertas_norm=None):
    df = df_ventas.copy()
    if COL_FECHA not in df.columns:
        return pd.DataFrame()

    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_FECHA])
    df["Anio"] = df[COL_FECHA].dt.year
    df["Mes"] = df[COL_FECHA].dt.month

    mensual = df.groupby([COL_CN, "Anio", "Mes"])[COL_VENTAS].sum().reset_index()

    mirror = mensual.copy().rename(columns={COL_VENTAS: "Base_Mirroring"})
    mirror["Anio"] = mirror["Anio"] + 1
    mensual = mensual.merge(mirror[[COL_CN, "Anio", "Mes", "Base_Mirroring"]],
                            on=[COL_CN, "Anio", "Mes"], how="left")
    mensual["Base_Mirroring"] = mensual["Base_Mirroring"].fillna(0)

    ventas_anual = mensual.groupby([COL_CN, "Anio"])[COL_VENTAS].sum().reset_index()
    growth = {}
    for cn, g in ventas_anual.groupby(COL_CN):
        vals = g.sort_values("Anio")[COL_VENTAS].values
        if len(vals) >= 2 and vals[-2] > 0:
            growth[cn] = np.clip(vals[-1] / vals[-2] - 1, -0.5, 0.5)
        else:
            growth[cn] = 0.0
    mensual["Growth_Factor"] = mensual[COL_CN].map(growth).fillna(0)

    mensual["Lag_30"] = mensual.groupby(COL_CN)[COL_VENTAS].shift(1).fillna(0)
    mensual["Pct_Zona_Cobro"] = mensual.apply(
        lambda r: _pct_zona_cobro(r["Anio"], r["Mes"]), axis=1)
    mensual["Es_Paga_Extra"] = mensual["Mes"].isin([6, 12]).astype(int)

    mensual["Horas_Abierto"] = mensual.apply(
        lambda r: calcular_horas_mes(int(r["Anio"]), int(r["Mes"]), calendario), axis=1)
    mensual["Dias_Abiertos"] = mensual.apply(
        lambda r: calcular_dias_abiertos_mes(int(r["Anio"]), int(r["Mes"]), calendario), axis=1)

    mensual["Epi_Gripe"] = perfil.get("epi_gripe", 0)
    mensual["Epi_Alergias"] = perfil.get("epi_alergias", 0)
    mensual["Epi_Covid"] = perfil.get("epi_covid", 0)

    for key in ["centro_salud", "residencia", "colegio", "zona_turistica", "zona_rural"]:
        mensual[f"Perfil_{key}"] = int(perfil.get(key, False))

    mensual["Mes_Num"] = mensual["Mes"]

    cns_promo = obtener_cns_con_promo_historica()
    mensual["Was_Promo"] = mensual[COL_CN].isin(cns_promo).astype(int)

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

    atc_map = _obtener_mapa_atc(df_inventario)
    mensual["_grupo_atc"] = mensual[COL_CN].map(atc_map).fillna("OTRO")
    mensual = mensual.sort_values([COL_CN, "Anio", "Mes"]).reset_index(drop=True)
    global_mean_ventas = mensual[COL_VENTAS].mean()
    atc_cumsum = mensual.groupby("_grupo_atc")[COL_VENTAS].cumsum() - mensual[COL_VENTAS]
    atc_cumcount = mensual.groupby("_grupo_atc").cumcount()
    mensual["ATC_Encoded"] = np.where(atc_cumcount > 0, atc_cumsum / atc_cumcount, global_mean_ventas)
    mensual["ATC_Encoded"] = mensual["ATC_Encoded"].round(2)
    mensual = mensual.drop(columns=["_grupo_atc"])

    temp_data = st.session_state.get("temperatura_historica")
    zona = perfil.get("zona_climatica", "mediterraneo")
    mensual["Temp_Media"] = mensual.apply(
        lambda r: _obtener_temperatura(int(r["Anio"]), int(r["Mes"]), temp_data, zona), axis=1)
    if temp_data is not None and not temp_data.empty:
        temp_clima = mensual.groupby("Mes")["Temp_Media"].mean().to_dict()
        mensual["Temp_Desviacion"] = mensual.apply(
            lambda r: round(r["Temp_Media"] - temp_clima.get(r["Mes"], r["Temp_Media"]), 1), axis=1)
    else:
        mensual["Temp_Desviacion"] = 0.0

    if COL_STOCK in df_inventario.columns:
        stock_map = df_inventario.set_index(COL_CN)[COL_STOCK].to_dict()
        vm_global = mensual.groupby(COL_CN)[COL_VENTAS].mean().to_dict()
        ultimo_idx = mensual.groupby(COL_CN).tail(1).index
        mensual["Stock_Ratio"] = 0.0
        for _sr_idx in ultimo_idx:
            _sr_cn = mensual.at[_sr_idx, COL_CN]
            mensual.at[_sr_idx, "Stock_Ratio"] = round(safe_div(stock_map.get(_sr_cn, 0), vm_global.get(_sr_cn, 1)), 2)
    else:
        mensual["Stock_Ratio"] = 0.0

    return mensual

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

def obtener_modelo_cacheado():
    if "modelo_ml_cache" not in st.session_state:
        model, met, rmse = cargar_modelo_farmacia()
        st.session_state["modelo_ml_cache"] = (model, met, rmse)
    return st.session_state["modelo_ml_cache"]

def invalidar_cache_modelo():
    st.session_state.pop("modelo_ml_cache", None)

def necesita_reentrenamiento():
    _, met, _ = obtener_modelo_cacheado()
    if met is None:
        return False
    n_hist_modelo = met.get("n_registros_historico", 0)
    df_hist = st.session_state.get("historico")
    if df_hist is None:
        return False
    return len(df_hist) != n_hist_modelo

def predecir_demanda_ensemble(model, df_features_futuro, rmse_por_cn,
                               df_ventas_media_heuristico, nivel_servicio_pct=95):
    pred_ml = predecir_demanda_ml(model, df_features_futuro, rmse_por_cn, nivel_servicio_pct)
    if df_ventas_media_heuristico is None or df_ventas_media_heuristico.empty:
        return pred_ml
    heur = df_ventas_media_heuristico[[COL_CN, "Venta_Media_Mensual"]].copy()
    heur = heur.rename(columns={"Venta_Media_Mensual": "Pred_Heuristico"})
    result = pred_ml.merge(heur, on=COL_CN, how="left")
    result["Pred_Heuristico"] = result["Pred_Heuristico"].fillna(result["Prediccion_Media"])
    if "Venta_Std_Mensual" in df_ventas_media_heuristico.columns:
        std_map = df_ventas_media_heuristico.set_index(COL_CN)["Venta_Std_Mensual"].to_dict()
        result["_std"] = result[COL_CN].map(std_map).fillna(result["RMSE_Producto"])
    else:
        result["_std"] = result["RMSE_Producto"]
    std_clipped = result["_std"].clip(lower=0.1)
    result["_peso_ml"] = (1 - result["RMSE_Producto"] / std_clipped).clip(0.2, 0.8)
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

def generar_pedido_ensemble(df_inventario, model, df_features_futuro, rmse_por_cn,
                            df_ventas_media_heuristico, meses_cobertura,
                            nivel_servicio_pct, df_ofertas=None, productos_protegidos=None):
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
