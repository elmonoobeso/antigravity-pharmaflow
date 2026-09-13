import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import pandas as pd
import numpy as np
from datetime import date, datetime
from calendar import monthrange
from data.io import cargar_json_farmacia, guardar_json_farmacia, ruta_farmacia_activa
from utils.helpers import normalizar_cn
from config.settings import (
    COL_CN, COL_VENTAS, COL_FECHA, COL_MOLECULA, Z_SCORES
)
from core.business import (
    calcular_horas_mes, calcular_dias_abiertos_mes,
    generar_pedido_cobertura, crecimiento_12m
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

# Variables agrupadas por fuente. La seleccion automatica decide por farmacia que
# grupos se quedan; una fuente nueva (p.ej. incidencia de gripe semanal) se anade
# registrando aqui su grupo. Perfil y nivel epidemiologico manual no estan: son un
# valor fijo en todo el historico de la farmacia y el modelo no puede aprender de ellos.
GRUPO_BASE = "Historia de ventas"
GRUPOS_FEATURES = {
    GRUPO_BASE: ["Base_Mirroring", "Lag_30", "Growth_Factor"],
    "Calendario": ["Mes_Num", "Pct_Zona_Cobro", "Es_Paga_Extra", "Horas_Abierto", "Dias_Abiertos"],
    "Clima": ["Temp_Media", "Temp_Desviacion"],
    "Grupo terapeutico": ["ATC_Encoded"],
    "Promociones": ["Was_Promo", "Is_Future_Promo"],
}
FEATURE_COLS = [c for cols in GRUPOS_FEATURES.values() for c in cols]

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
            return {normalizar_cn(k): str(v)[:5] for k, v in mapa.items() if pd.notna(v)}
    if COL_MOLECULA in df_inventario.columns:
        mol_map = df_inventario.set_index(COL_CN)[COL_MOLECULA].to_dict()
        grupo_map = {}
        for cn, mol in mol_map.items():
            mol_str = str(mol).strip().lower()
            if mol_str in ("parafarmacia", "", "nan"):
                grupo_map[normalizar_cn(cn)] = "PARA"
            else:
                grupo_map[normalizar_cn(cn)] = mol_str[:20]
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

MIN_MESES_COLD_START = 6
MESES_DESFASE_AVISO = 2

def _columnas_modelo(model, df):
    """Variables con las que se entreno el modelo (la seleccion puede haber quitado grupos)."""
    nombres = getattr(model, "feature_names_in_", None)
    if nombres is not None:
        return list(nombres)
    return [c for c in FEATURE_COLS if c in df.columns]

def _agregar_mensual(df_ventas):
    """Historico de ventas -> una fila por producto y mes (Anio, Mes, Ventas)."""
    if df_ventas is None or COL_FECHA not in df_ventas.columns:
        return pd.DataFrame()
    df = df_ventas.copy()
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_FECHA])
    df["Anio"] = df[COL_FECHA].dt.year.astype(int)
    df["Mes"] = df[COL_FECHA].dt.month.astype(int)
    return df.groupby([COL_CN, "Anio", "Mes"])[COL_VENTAS].sum().reset_index()

def _promos_por_periodo():
    """(CN, periodo) con promocion registrada en ese mes. Antes Was_Promo marcaba
    todas las filas de un producto con cualquier promo registrada, aunque fuera
    posterior al mes (informacion del futuro)."""
    res = set()
    for r in cargar_json_farmacia("historico_promociones.json", default=[]):
        fecha = pd.to_datetime(r.get("fecha"), errors="coerce")
        cn = normalizar_cn(r.get("codigo_nacional"))
        if cn and pd.notna(fecha):
            res.add((cn, fecha.year * 12 + fecha.month))
    return res

def _features_desde_mensual(mensual, df_inventario, perfil, calendario, df_ofertas_norm=None):
    mensual = mensual.sort_values([COL_CN, "Anio", "Mes"]).reset_index(drop=True)
    mensual["_periodo"] = mensual["Anio"] * 12 + mensual["Mes"]
    ventas_p = mensual[[COL_CN, "_periodo", COL_VENTAS]]

    # Mismo mes del ano anterior y mes anterior, por calendario (0 si no hubo ventas).
    previo = ventas_p.assign(_periodo=ventas_p["_periodo"] + 12).rename(columns={COL_VENTAS: "Base_Mirroring"})
    lag = ventas_p.assign(_periodo=ventas_p["_periodo"] + 1).rename(columns={COL_VENTAS: "Lag_30"})
    mensual = mensual.merge(previo, on=[COL_CN, "_periodo"], how="left").merge(lag, on=[COL_CN, "_periodo"], how="left")
    mensual[["Base_Mirroring", "Lag_30"]] = mensual[["Base_Mirroring", "Lag_30"]].fillna(0)

    # Crecimiento 12m vs 12m previos calculado en cada mes solo con meses anteriores:
    # la version previa usaba los dos ultimos anos naturales para todas las filas (fuga).
    crec = crecimiento_12m(ventas_p).rename(columns={"Crecimiento_12m": "Growth_Factor"})
    mensual = mensual.merge(crec, on=[COL_CN, "_periodo"], how="left")
    mensual["Growth_Factor"] = mensual["Growth_Factor"].fillna(0.0)

    # Variables de calendario y clima: una vez por mes, no por fila.
    temp_data = st.session_state.get("temperatura_historica")
    zona = perfil.get("zona_climatica", "mediterraneo")
    cal_rows = []
    for anio, mes in mensual[["Anio", "Mes"]].drop_duplicates().itertuples(index=False):
        anio, mes = int(anio), int(mes)
        cal_rows.append({
            "Anio": anio, "Mes": mes,
            "Pct_Zona_Cobro": _pct_zona_cobro(anio, mes),
            "Horas_Abierto": calcular_horas_mes(anio, mes, calendario),
            "Dias_Abiertos": calcular_dias_abiertos_mes(anio, mes, calendario),
            "Temp_Media": _obtener_temperatura(anio, mes, temp_data, zona),
        })
    mensual = mensual.merge(pd.DataFrame(cal_rows), on=["Anio", "Mes"], how="left")
    mensual["Es_Paga_Extra"] = mensual["Mes"].isin([6, 12]).astype(int)

    mensual["Epi_Gripe"] = perfil.get("epi_gripe", 0)
    mensual["Epi_Alergias"] = perfil.get("epi_alergias", 0)
    mensual["Epi_Covid"] = perfil.get("epi_covid", 0)
    for key in ["centro_salud", "residencia", "colegio", "zona_turistica", "zona_rural"]:
        mensual[f"Perfil_{key}"] = int(perfil.get(key, False))

    mensual["Mes_Num"] = mensual["Mes"]

    promos = _promos_por_periodo()
    mensual["Was_Promo"] = [int((cn, p) in promos) for cn, p in zip(mensual[COL_CN], mensual["_periodo"], strict=True)]

    cns_oferta_actual = set()
    if df_ofertas_norm is not None and not df_ofertas_norm.empty:
        cns_oferta_actual = set(df_ofertas_norm["Oferta_Nombre"].str.lower())
    if COL_MOLECULA in df_inventario.columns:
        mol_map = df_inventario.set_index(COL_CN)[COL_MOLECULA].to_dict()
        mensual["Is_Future_Promo"] = mensual[COL_CN].map(mol_map).fillna("").str.lower().isin(cns_oferta_actual).astype(int)
    else:
        mensual["Is_Future_Promo"] = 0

    atc_map = _obtener_mapa_atc(df_inventario)
    mensual["_grupo_atc"] = mensual[COL_CN].map(atc_map).fillna("OTRO")
    mensual["ATC_Encoded"] = _codificar_atc_sin_fuga(mensual)
    mensual = mensual.drop(columns=["_grupo_atc"])

    if temp_data is not None and not temp_data.empty:
        temp_clima = mensual.groupby("Mes")["Temp_Media"].mean()
        mensual["Temp_Desviacion"] = (mensual["Temp_Media"] - mensual["Mes"].map(temp_clima)).round(1)
    else:
        mensual["Temp_Desviacion"] = 0.0

    # Stock_Ratio se elimino como feature: solo existe inventario actual, no
    # snapshots historicos, asi que en entrenamiento valia 0 en casi todas las
    # filas y en prediccion tomaba un valor que el modelo nunca habia visto. El
    # stock real ya se descuenta despues, al calcular las unidades del pedido.
    return mensual.drop(columns=["_periodo"])

def build_features(df_ventas, df_inventario, perfil, calendario, df_ofertas_norm=None):
    mensual = _agregar_mensual(df_ventas)
    if mensual.empty:
        return pd.DataFrame()
    return _features_desde_mensual(mensual, df_inventario, perfil, calendario, df_ofertas_norm)

def _proxy_cold_start(mensual, df_inventario):
    """Meses de historico por CN y venta mensual media de su molecula (productos
    maduros, ultimos 12 meses). Solo ajusta predicciones: nunca las ventas reales
    con las que se entrena y se mide el modelo."""
    n_meses = mensual.groupby(COL_CN).size()
    if COL_MOLECULA not in df_inventario.columns:
        return n_meses, pd.Series(dtype=float)
    periodo = mensual["Anio"] * 12 + mensual["Mes"]
    recientes = mensual[periodo > periodo.max() - 12]
    maduros = recientes[recientes[COL_CN].map(n_meses) >= MIN_MESES_COLD_START]
    mol_map = df_inventario.set_index(COL_CN)[COL_MOLECULA].to_dict()
    media_mol = maduros.assign(_mol=maduros[COL_CN].map(mol_map)).groupby("_mol")[COL_VENTAS].mean()
    proxy = pd.Series({cn: media_mol.get(mol_map.get(cn), np.nan) for cn in n_meses.index}, dtype=float)
    return n_meses, proxy

def construir_features_futuras(model, df_ventas, df_inventario, perfil, calendario,
                               df_ofertas_norm=None, meses_cobertura=2, hoy=None):
    """Features de los meses que cubre el pedido: del mes siguiente a hoy en adelante.

    Los meses entre el final del historico y el inicio de la cobertura se predicen
    de forma recursiva: la prediccion de un mes alimenta el Lag, el Mirroring y el
    crecimiento del siguiente. Productos con menos de MIN_MESES_COLD_START meses de
    historico mezclan 50/50 la prediccion con la media de su molecula.
    Devuelve (df_futuro con columna Prediccion_Base, meses sin datos reales).
    """
    mensual = _agregar_mensual(df_ventas)
    if mensual.empty:
        return pd.DataFrame(), 0
    hoy = hoy or date.today()
    mensual[COL_VENTAS] = mensual[COL_VENTAS].astype(float)
    n_meses, proxy = _proxy_cold_start(mensual, df_inventario)
    p_ultimo = int((mensual["Anio"] * 12 + mensual["Mes"]).max())
    p_ini = max(hoy.year * 12 + hoy.month + 1, p_ultimo + 1)
    p_fin = p_ini + int(meses_cobertura) - 1
    cns = mensual[COL_CN].unique()
    futuras = []
    for p in range(p_ultimo + 1, p_fin + 1):
        anio, mes = (p - 1) // 12, (p - 1) % 12 + 1
        mensual = pd.concat([mensual, pd.DataFrame({COL_CN: cns, "Anio": anio, "Mes": mes, COL_VENTAS: 0.0})],
                            ignore_index=True)
        feats = _features_desde_mensual(mensual, df_inventario, perfil, calendario, df_ofertas_norm)
        filas = feats[(feats["Anio"] == anio) & (feats["Mes"] == mes)].copy()
        pred = model.predict(filas[_columnas_modelo(model, filas)].fillna(0)).clip(min=0)
        nuevo = (filas[COL_CN].map(n_meses).fillna(0) < MIN_MESES_COLD_START) & filas[COL_CN].map(proxy).notna()
        filas["Prediccion_Base"] = np.where(nuevo, 0.5 * pred + 0.5 * filas[COL_CN].map(proxy).fillna(0), pred)
        mask = (mensual["Anio"] == anio) & (mensual["Mes"] == mes)
        pred_map = dict(zip(filas[COL_CN], filas["Prediccion_Base"], strict=True))
        mensual.loc[mask, COL_VENTAS] = mensual.loc[mask, COL_CN].map(pred_map)
        if p >= p_ini:
            futuras.append(filas)
    return pd.concat(futuras, ignore_index=True), p_ini - 1 - p_ultimo

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

def _rmse_validacion_temporal(df_train, cols, params, n_folds=MESES_VALIDACION):
    """RMSE medio prediciendo los ultimos meses del train, cada uno entrenando solo con los anteriores."""
    periodos = sorted(_periodos(df_train).unique())
    folds = periodos[-n_folds:] if len(periodos) > n_folds + 2 else periodos[-1:]
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
    return (float(np.mean(errores)), len(errores)) if errores else (None, 0)

def _ajustar_hiperparametros(df_train, cols, n_folds=MESES_VALIDACION):
    """Elige hiperparametros con validacion temporal DENTRO del periodo de train.

    El holdout final no participa: si se eligiera la configuracion mirando el
    test, el test dejaria de ser una medida honesta.
    """
    ensayos = []
    for params in REJILLA_HIPERPARAMETROS:
        rmse, n = _rmse_validacion_temporal(df_train, cols, params, n_folds)
        if rmse is not None:
            ensayos.append({"params": params, "rmse_validacion": round(rmse, 3), "n_folds": n})
    if not ensayos:
        return REJILLA_HIPERPARAMETROS[1], []
    ensayos.sort(key=lambda e: e["rmse_validacion"])
    return ensayos[0]["params"], ensayos

def _seleccionar_grupos(df_train, params):
    """Seleccion hacia atras por grupos, con la misma validacion temporal del train.

    Se reentrena quitando cada grupo: si el error de validacion no empeora, el grupo
    se descarta (a igualdad, el modelo mas simple). La historia de ventas es
    obligatoria. El holdout no participa en la decision.
    """
    disponibles = {g: [c for c in vs if c in df_train.columns] for g, vs in GRUPOS_FEATURES.items()}
    todas = [c for vs in disponibles.values() for c in vs]
    rmse_todas, _ = _rmse_validacion_temporal(df_train, todas, params)
    if rmse_todas is None or rmse_todas <= 0:
        return todas, {}
    grupos, descartados = [], []
    for grupo, vs in disponibles.items():
        if not vs:
            continue
        rmse_sin, _ = _rmse_validacion_temporal(df_train, [c for c in todas if c not in vs], params)
        if rmse_sin is None:
            continue
        impacto = (rmse_sin - rmse_todas) / rmse_todas * 100
        if grupo == GRUPO_BASE:
            decision = "obligatorio"
        elif impacto > 0:
            decision = "se mantiene"
        else:
            decision = "se descarta"
            descartados.append(grupo)
        grupos.append({"grupo": grupo, "variables": vs, "rmse_sin_grupo": round(rmse_sin, 3),
                       "impacto_pct": round(impacto, 2), "decision": decision})
    elegidas = [c for g, vs in disponibles.items() if g not in descartados for c in vs]
    rmse_elegidas, _ = _rmse_validacion_temporal(df_train, elegidas, params)
    nota = None
    if rmse_elegidas is not None and rmse_elegidas > rmse_todas:
        # Quitar varios grupos a la vez puede empeorar aunque cada uno por separado no lo haga.
        elegidas, rmse_elegidas = todas, rmse_todas
        nota = "Quitar a la vez los grupos descartados empeoraba el error: se mantienen todas las variables."
        for g in grupos:
            if g["decision"] == "se descarta":
                g["decision"] = "se mantiene"
    return elegidas, {"rmse_todas": round(rmse_todas, 3), "rmse_elegidas": round(rmse_elegidas, 3),
                      "grupos": grupos, "nota": nota}

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
            "train_desde": _fmt_periodo(periodos[0]),
            "train_hasta": _fmt_periodo(p_test - 1),
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

def _backtest_horizonte(modelo, contexto, p_corte, p_ultimo):
    """Mide la prediccion tal como la usa el pedido.

    Desde el ultimo mes de train se predicen los meses del holdout encadenados (el
    mes 2 usa la prediccion del mes 1), sin ver ninguna venta real posterior al
    corte. El modelo es el del holdout: no ha visto esos meses ni para elegir
    variables ni hiperparametros. Devuelve el error por mes de horizonte y el del
    total del periodo por producto, que es lo que decide las unidades a pedir.
    """
    df_v = contexto["df_ventas"]
    fechas = pd.to_datetime(df_v[COL_FECHA], errors="coerce", dayfirst=True)
    df_hist = df_v[(fechas.dt.year * 12 + fechas.dt.month) < p_corte]
    p_origen = p_corte - 1
    n_meses = int(p_ultimo - p_corte + 1)
    hoy = date((p_origen - 1) // 12, (p_origen - 1) % 12 + 1, 15)
    futuro, _ = construir_features_futuras(modelo, df_hist, contexto["df_inventario"], contexto["perfil"],
                                           contexto["calendario"], contexto.get("df_ofertas"), n_meses, hoy=hoy)
    if futuro.empty:
        return {}
    real = _agregar_mensual(df_v)
    real["_p"] = real["Anio"] * 12 + real["Mes"]
    futuro["_p"] = futuro["Anio"] * 12 + futuro["Mes"]
    m = futuro.merge(real[[COL_CN, "_p", COL_VENTAS]].rename(columns={COL_VENTAS: "_real"}),
                     on=[COL_CN, "_p"], how="left")
    m["_real"] = m["_real"].fillna(0.0)
    pasos = []
    for h, (p, g) in enumerate(m.groupby("_p"), start=1):
        pasos.append({"h": h, "periodo": _fmt_periodo(p),
                      "ml": _metricas_error(g["_real"], g["Prediccion_Base"]),
                      "baseline": _metricas_error(g["_real"], g["Base_Mirroring"])})
    tot = m.groupby(COL_CN)[["_real", "Prediccion_Base", "Base_Mirroring"]].sum()
    ml_t = _metricas_error(tot["_real"], tot["Prediccion_Base"])
    base_t = _metricas_error(tot["_real"], tot["Base_Mirroring"])
    mejora = round((base_t["rmse"] - ml_t["rmse"]) / base_t["rmse"] * 100, 1) if base_t["rmse"] > 0 else None
    return {"origen": _fmt_periodo(p_origen), "n_meses": n_meses, "pasos": pasos,
            "total": {"ml": ml_t, "baseline": base_t, "mejora_pct": mejora}}

def entrenar_modelo_ml(df_features, contexto=None):
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
    # 2b. Seleccion automatica de grupos de variables, tambien solo con train.
    cols, seleccion = _seleccionar_grupos(df_train, mejores_params)

    # 3. Holdout: se mide una vez, sin haber influido en ninguna decision.
    modelo_holdout = _construir_modelo(mejores_params)
    modelo_holdout.fit(df_train[cols].fillna(0), df_train[COL_VENTAS])
    pred_holdout = modelo_holdout.predict(df_test[cols].fillna(0)).clip(min=0)
    met_holdout = _metricas_error(df_test[COL_VENTAS], pred_holdout)
    met_holdout["r2"] = round(float(r2_score(df_test[COL_VENTAS], pred_holdout)), 4)
    met_baseline_holdout = _metricas_error(df_test[COL_VENTAS], _prediccion_baseline(df_test))
    # 3b. Holdout a varios meses vista, encadenado como en el pedido (necesita el historico).
    backtest_horizonte = _backtest_horizonte(modelo_holdout, contexto, p_corte, periodos[-1]) if contexto else {}

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
        "features": cols, "importance": imp, "seleccion_variables": seleccion,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_registros_historico": len(df_features),
        "validacion": "split temporal + backtest walk-forward",
        "hiperparametros": mejores_params,
        "ensayos_tuning": ensayos,
        "holdout": met_holdout,
        "holdout_baseline": met_baseline_holdout,
        "backtest_horizonte": backtest_horizonte,
        "backtest": backtest,
        "peores_productos": peores,
        "residuos_hist": hist_residuos,
        "n_periodos": len(periodos),
        "periodo_corte": _fmt_periodo(p_corte),
    }
    return modelo_final, metricas, rmse_por_cn

def predecir_demanda_ml(model, df_features_futuro, rmse_por_cn, nivel_servicio_pct=95):
    if "Prediccion_Base" in df_features_futuro.columns:
        predicciones = df_features_futuro["Prediccion_Base"].to_numpy(dtype=float)
    else:
        predicciones = model.predict(df_features_futuro[_columnas_modelo(model, df_features_futuro)].fillna(0)).clip(min=0)
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
            "serie_mensual": [
                {"periodo": _fmt_periodo(pp), "real": round(float(v), 1)}
                for pp, v in df_features.groupby(_periodos(df_features))[COL_VENTAS].sum().items()
            ] if periodos else [],
        },
        "features": {
            "usadas": metricas.get("features", []),
            "importancia": metricas.get("importance", {}),
            "seleccion": metricas.get("seleccion_variables", {}),
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
        "backtest_horizonte": metricas.get("backtest_horizonte", {}),
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
    df_vm_ml = pred[[COL_CN, "Prediccion_Media", "RMSE_Producto"]].copy()
    df_vm_ml = df_vm_ml.rename(columns={
        "Prediccion_Media": "Venta_Media_Mensual",
        "RMSE_Producto": "Venta_Std_Mensual",
    })
    return generar_pedido_cobertura(
        df_inventario, df_vm_ml, meses_cobertura, df_ofertas,
        nivel_servicio=Z_SCORES.get(nivel_servicio_pct, 1.645),
        productos_protegidos=productos_protegidos)

def generar_pedido_ensemble(df_inventario, model, df_features_futuro, rmse_por_cn,
                            df_ventas_media_heuristico, meses_cobertura,
                            nivel_servicio_pct, df_ofertas=None, productos_protegidos=None):
    pred = predecir_demanda_ensemble(
        model, df_features_futuro, rmse_por_cn,
        df_ventas_media_heuristico, nivel_servicio_pct)
    df_vm = pred[[COL_CN, "Prediccion_Media", "RMSE_Producto"]].copy()
    df_vm = df_vm.rename(columns={
        "Prediccion_Media": "Venta_Media_Mensual",
        "RMSE_Producto": "Venta_Std_Mensual",
    })
    return generar_pedido_cobertura(
        df_inventario, df_vm, meses_cobertura, df_ofertas,
        nivel_servicio=Z_SCORES.get(nivel_servicio_pct, 1.645),
        productos_protegidos=productos_protegidos)
