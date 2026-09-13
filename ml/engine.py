import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import pandas as pd
import numpy as np
from datetime import datetime
from calendar import monthrange
from data.io import cargar_json_farmacia, guardar_json_farmacia, ruta_farmacia_activa
from config.settings import (
    COL_CN, COL_VENTAS, COL_FECHA, COL_MOLECULA, Z_SCORES
)
from core.business import (
    calcular_horas_mes, calcular_dias_abiertos_mes, obtener_cns_con_promo_historica,
    generar_pedido_cobertura
)
import streamlit as st

try:
    from xgboost import XGBRegressor
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
    "ATC_Encoded", "Temp_Media", "Temp_Desviacion",
]

# Meses reservados como holdout final. No se usan ni para entrenar ni para
# elegir hiperparametros: solo para medir.
MESES_TEST_HOLDOUT = 3
MESES_VALIDACION = 3
MIN_PERIODOS_HISTORICO = 12

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

def _codificar_atc_sin_fuga(mensual):
    """Media historica de ventas del grupo ATC usando SOLO meses anteriores.

    La version previa hacia cumsum sobre un dataframe ordenado por CN, no por
    fecha: para un producto con historico antiguo, la media acumulada de su
    grupo ya incluia ventas de otros productos en meses posteriores. Es decir,
    el modelo veia el futuro. Aqui se agrega por (grupo, mes) y se toma la media
    expansiva desplazada un periodo, de modo que cada fila solo ve el pasado.
    """
    periodo = mensual["Anio"] * 12 + mensual["Mes"]
    agg = (mensual.assign(_periodo=periodo)
                  .groupby(["_grupo_atc", "_periodo"])[COL_VENTAS].mean()
                  .reset_index(name="_media_periodo")
                  .sort_values(["_grupo_atc", "_periodo"]))
    agg["_enc"] = (agg.groupby("_grupo_atc")["_media_periodo"]
                      .transform(lambda s: s.shift(1).expanding().mean()))

    # Respaldo para el primer mes de cada grupo: media global de meses anteriores.
    glob = (mensual.assign(_periodo=periodo)
                   .groupby("_periodo")[COL_VENTAS].mean()
                   .sort_index())
    glob_exp = glob.shift(1).expanding().mean()

    enc = mensual.assign(_periodo=periodo).merge(
        agg[["_grupo_atc", "_periodo", "_enc"]], on=["_grupo_atc", "_periodo"], how="left")
    resultado = enc["_enc"].fillna(enc["_periodo"].map(glob_exp)).fillna(0.0)
    return resultado.round(2).values

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
    mensual["ATC_Encoded"] = _codificar_atc_sin_fuga(mensual)
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

    # Stock_Ratio se elimino como feature: solo existe inventario actual, no
    # snapshots historicos, asi que en entrenamiento valia 0 en casi todas las
    # filas y en prediccion tomaba un valor que el modelo nunca habia visto. El
    # stock real ya se descuenta despues, al calcular las unidades del pedido.
    return mensual

REJILLA_HIPERPARAMETROS = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 200},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.10, "n_estimators": 200},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 6, "learning_rate": 0.10, "n_estimators": 200},
]

def _periodos(df):
    """Serie entera Anio*12+Mes: ordena meses sin depender del formato de fecha."""
    return df["Anio"].astype(int) * 12 + df["Mes"].astype(int)

def _fmt_periodo(p):
    """Convierte el entero de periodo a 'AAAA-MM'. Diciembre es el caso borde:
    p % 12 vale 0 y p // 12 ya ha sumado un anio, de ahi el -1."""
    p = int(p)
    return f"{(p - 1) // 12}-{(p - 1) % 12 + 1:02d}"

def _construir_modelo(params):
    return XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42,
                        verbosity=0, n_jobs=-1, **params)

def _metricas_error(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_real
    no_cero = y_real > 0
    mape = float(np.mean(np.abs(err[no_cero] / y_real[no_cero])) * 100) if no_cero.any() else None
    return {
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "mape": round(mape, 1) if mape is not None else None,
        "sesgo": round(float(np.mean(err)), 2),
        "n": int(len(y_real)),
    }

def _prediccion_baseline(df_fold):
    """Baseline honesto: mismo mes del ano pasado y, si no existe, el mes previo.

    Es el comparador contra el que el ML tiene que demostrar que aporta algo.
    """
    mirror = df_fold["Base_Mirroring"].astype(float) if "Base_Mirroring" in df_fold.columns else pd.Series(0.0, index=df_fold.index)
    lag = df_fold["Lag_30"].astype(float) if "Lag_30" in df_fold.columns else pd.Series(0.0, index=df_fold.index)
    return mirror.where(mirror > 0, lag).values

def _ajustar_hiperparametros(df_train, cols, n_folds=MESES_VALIDACION):
    """Elige hiperparametros con validacion temporal DENTRO del periodo de train.

    El holdout final no participa: si se eligiera la configuracion mirando el
    test, el test dejaria de ser una medida honesta.
    """
    periodos = sorted(_periodos(df_train).unique())
    folds = periodos[-n_folds:] if len(periodos) > n_folds + 2 else periodos[-1:]
    ensayos = []
    for params in REJILLA_HIPERPARAMETROS:
        errores = []
        for p_val in folds:
            mask_tr = _periodos(df_train) < p_val
            mask_val = _periodos(df_train) == p_val
            if mask_tr.sum() < 10 or mask_val.sum() == 0:
                continue
            modelo = _construir_modelo(params)
            modelo.fit(df_train.loc[mask_tr, cols].fillna(0), df_train.loc[mask_tr, COL_VENTAS])
            pred = modelo.predict(df_train.loc[mask_val, cols].fillna(0)).clip(min=0)
            errores.append(np.sqrt(mean_squared_error(df_train.loc[mask_val, COL_VENTAS], pred)))
        if errores:
            ensayos.append({"params": params, "rmse_validacion": round(float(np.mean(errores)), 3),
                            "n_folds": len(errores)})
    if not ensayos:
        return REJILLA_HIPERPARAMETROS[1], []
    ensayos.sort(key=lambda e: e["rmse_validacion"])
    return ensayos[0]["params"], ensayos

def _backtest_walk_forward(df, cols, params, n_folds=MESES_TEST_HOLDOUT):
    """Reentrena avanzando mes a mes y predice el siguiente, como en produccion.

    Devuelve las metricas por fold, el comparador baseline y el error por
    producto, que es lo que alimenta el stock de seguridad.
    """
    periodos = sorted(_periodos(df).unique())
    folds_periodos = periodos[-n_folds:]
    detalle, residuos = [], []
    for p_test in folds_periodos:
        mask_tr = _periodos(df) < p_test
        mask_te = _periodos(df) == p_test
        if mask_tr.sum() < 10 or mask_te.sum() == 0:
            continue
        modelo = _construir_modelo(params)
        modelo.fit(df.loc[mask_tr, cols].fillna(0), df.loc[mask_tr, COL_VENTAS])
        y_real = df.loc[mask_te, COL_VENTAS].values
        y_pred = modelo.predict(df.loc[mask_te, cols].fillna(0)).clip(min=0)
        y_base = _prediccion_baseline(df.loc[mask_te])
        detalle.append({
            "periodo": _fmt_periodo(p_test),
            "n_train": int(mask_tr.sum()),
            "ml": _metricas_error(y_real, y_pred),
            "baseline": _metricas_error(y_real, y_base),
            "total_real": round(float(np.sum(y_real)), 1),
            "total_ml": round(float(np.sum(y_pred)), 1),
            "total_baseline": round(float(np.sum(y_base)), 1),
        })
        residuos.append(pd.DataFrame({COL_CN: df.loc[mask_te, COL_CN].values,
                                      "y_real": y_real, "y_pred": y_pred, "y_base": y_base}))
    if not detalle:
        return {"folds": [], "ml": {}, "baseline": {}, "mejora_pct": None}, {}, pd.DataFrame()

    res = pd.concat(residuos, ignore_index=True)
    global_ml = _metricas_error(res["y_real"], res["y_pred"])
    global_base = _metricas_error(res["y_real"], res["y_base"])
    mejora = None
    if global_base["rmse"] > 0:
        mejora = round((global_base["rmse"] - global_ml["rmse"]) / global_base["rmse"] * 100, 1)
    rmse_por_cn = (res.assign(_e2=(res["y_real"] - res["y_pred"]) ** 2)
                      .groupby(COL_CN)["_e2"].mean().pow(0.5).round(3).to_dict())
    resumen = {"folds": detalle, "ml": global_ml, "baseline": global_base, "mejora_pct": mejora}
    return resumen, rmse_por_cn, res

def entrenar_modelo_ml(df_features):
    """Pipeline completo: split temporal -> tuning -> holdout -> backtest.

    Devuelve (modelo, metricas, rmse_por_cn). El rmse_por_cn sale del backtest
    walk-forward, no de un split aleatorio, porque de el depende el stock de
    seguridad que se recomienda al farmaceutico.
    """
    if not ML_AVAILABLE:
        return None, {"error": "XGBoost/sklearn no instalado"}, {}
    df = df_features.dropna(subset=[COL_VENTAS]).copy()
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(cols) < 3 or df.empty or "Anio" not in df.columns:
        return None, {"error": "Datos insuficientes para entrenar"}, {}

    df = df.sort_values(["Anio", "Mes", COL_CN]).reset_index(drop=True)
    periodos = sorted(_periodos(df).unique())
    if len(periodos) < MIN_PERIODOS_HISTORICO:
        return None, {"error": f"Se necesitan al menos {MIN_PERIODOS_HISTORICO} meses de historico "
                               f"(hay {len(periodos)}). Con menos, el error medido no es fiable."}, {}

    # 1. Split temporal: los ultimos meses quedan fuera de todo el ajuste.
    p_corte = periodos[-MESES_TEST_HOLDOUT]
    df_train = df[_periodos(df) < p_corte].copy()
    df_test = df[_periodos(df) >= p_corte].copy()
    if df_train.empty or df_test.empty:
        return None, {"error": "Historico insuficiente para reservar un holdout temporal"}, {}

    # 2. Hiperparametros por validacion temporal, solo con datos de train.
    mejores_params, ensayos = _ajustar_hiperparametros(df_train, cols)

    # 3. Holdout: se mide una vez, sin haber influido en ninguna decision.
    modelo_holdout = _construir_modelo(mejores_params)
    modelo_holdout.fit(df_train[cols].fillna(0), df_train[COL_VENTAS])
    pred_holdout = modelo_holdout.predict(df_test[cols].fillna(0)).clip(min=0)
    met_holdout = _metricas_error(df_test[COL_VENTAS], pred_holdout)
    met_holdout["r2"] = round(float(r2_score(df_test[COL_VENTAS], pred_holdout)), 4)
    met_baseline_holdout = _metricas_error(df_test[COL_VENTAS], _prediccion_baseline(df_test))

    # 4. Backtest walk-forward sobre todo el historico.
    backtest, rmse_por_cn, residuos = _backtest_walk_forward(df, cols, mejores_params)

    # 5. Modelo final: se reentrena con TODO, incluido el holdout ya medido.
    modelo_final = _construir_modelo(mejores_params)
    modelo_final.fit(df[cols].fillna(0), df[COL_VENTAS])
    imp = dict(sorted(zip(cols, (float(v) for v in modelo_final.feature_importances_)),
                      key=lambda x: x[1], reverse=True))

    peores, hist_residuos = [], {}
    if not residuos.empty:
        peores = (residuos.assign(_err=(residuos["y_pred"] - residuos["y_real"]).abs())
                          .groupby(COL_CN)["_err"].mean().nlargest(10).round(2)
                          .reset_index().to_dict("records"))
        err = (residuos["y_pred"] - residuos["y_real"]).values
        counts, bordes = np.histogram(err, bins=15)
        hist_residuos = {
            "counts": [int(c) for c in counts],
            "bordes": [round(float(b), 2) for b in bordes],
            "p50": round(float(np.percentile(err, 50)), 2),
            "p90": round(float(np.percentile(np.abs(err), 90)), 2),
        }

    metricas = {
        "rmse": backtest["ml"].get("rmse", met_holdout["rmse"]),
        "r2": met_holdout["r2"],
        "n_train": int(len(df_train)), "n_test": int(len(df_test)),
        "features": cols, "importance": imp,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_registros_historico": len(df_features),
        "validacion": "split temporal + backtest walk-forward",
        "hiperparametros": mejores_params,
        "ensayos_tuning": ensayos,
        "holdout": met_holdout,
        "holdout_baseline": met_baseline_holdout,
        "backtest": backtest,
        "peores_productos": peores,
        "residuos_hist": hist_residuos,
        "n_periodos": len(periodos),
        "periodo_corte": _fmt_periodo(p_corte),
    }
    return modelo_final, metricas, rmse_por_cn

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

def construir_artefacto_pipeline(df_features, metricas):
    """Deja por escrito cada fase del proceso ML para poder mostrarla en la UI."""
    periodos = sorted(_periodos(df_features).unique()) if "Anio" in df_features.columns else []
    return {
        "fecha": metricas.get("fecha_entrenamiento"),
        "datos": {
            "n_filas": int(len(df_features)),
            "n_productos": int(df_features[COL_CN].nunique()) if COL_CN in df_features.columns else 0,
            "n_periodos": len(periodos),
            "desde": _fmt_periodo(periodos[0]) if periodos else None,
            "hasta": _fmt_periodo(periodos[-1]) if periodos else None,
        },
        "features": {
            "usadas": metricas.get("features", []),
            "importancia": metricas.get("importance", {}),
        },
        "split": {
            "estrategia": "temporal (los ultimos meses nunca se usan para ajustar)",
            "periodo_corte": metricas.get("periodo_corte"),
            "n_train": metricas.get("n_train"),
            "n_test": metricas.get("n_test"),
            "meses_holdout": MESES_TEST_HOLDOUT,
        },
        "tuning": {
            "metodo": f"validacion temporal de {MESES_VALIDACION} meses dentro del train",
            "elegidos": metricas.get("hiperparametros", {}),
            "ensayos": metricas.get("ensayos_tuning", []),
        },
        "holdout": metricas.get("holdout", {}),
        "holdout_baseline": metricas.get("holdout_baseline", {}),
        "backtest": metricas.get("backtest", {}),
        "peores_productos": metricas.get("peores_productos", []),
        "residuos_hist": metricas.get("residuos_hist", {}),
    }

def guardar_modelo_farmacia(model, metricas, rmse_por_cn, df_features=None):
    ruta = ruta_farmacia_activa()
    if ruta is None or model is None:
        return
    joblib.dump(model, ruta / "modelo_ml.joblib")
    guardar_json_farmacia("modelo_metricas.json", metricas)
    guardar_json_farmacia("modelo_rmse_cn.json", rmse_por_cn)
    if df_features is not None:
        guardar_json_farmacia("modelo_pipeline.json", construir_artefacto_pipeline(df_features, metricas))

def cargar_modelo_farmacia():
    ruta = ruta_farmacia_activa()
    if ruta is None:
        return None, None, None
    model_path = ruta / "modelo_ml.joblib"
    if not model_path.exists():
        return None, None, None
    try:
        model = joblib.load(model_path)
        # Un modelo guardado con features que ya no existen (p.ej. Stock_Ratio)
        # haria estallar model.predict con un feature_names mismatch. Se trata
        # como "sin modelo" para que la UI ofrezca reentrenar.
        # feature_names_in_ es un ndarray: nada de "or []" aqui, comparar un
        # array con or lanza ValueError y lo tragaria el except de abajo.
        esperadas = getattr(model, "feature_names_in_", None)
        esperadas = list(esperadas) if esperadas is not None else []
        if any(f not in FEATURE_COLS for f in esperadas):
            return None, {"obsoleto": True}, {}
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
