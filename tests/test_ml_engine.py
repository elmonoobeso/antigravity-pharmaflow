"""Banco de pruebas del motor ML con datos sinteticos.

Comprueba que el codigo esta bien construido sin depender de datos reales:
  1. Ninguna variable de un mes usa ventas de ese mes o posteriores (fugas).
  2. Con un patron conocido (tendencia + estacionalidad) el ML gana al baseline.
  3. Con ruido puro no baja del error minimo posible (si lo hiciera, veria el futuro).
  4. Con las ventas barajadas no aprende nada (si aprendiera, habria una fuga).
  5. La prediccion futura cubre los meses pedidos, sin huecos ni valores invalidos.
  6. El backtest a varios meses predice encadenado sin ver ventas reales posteriores.

Uso:  py -3.14 tests/test_ml_engine.py      (no escribe nada en farmacias/)
"""
import logging
import sys
import time
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml.engine as E  # noqa: E402
from config.settings import COL_CN, COL_VENTAS  # noqa: E402
from core.business import cargar_calendario_farmacia  # noqa: E402

PERFIL = {}
CALENDARIO = cargar_calendario_farmacia()  # sin farmacia activa devuelve el horario por defecto
PERIODO_INICIO = 2021 * 12 + 1  # enero 2021


def _anio_mes(p):
    return (p - 1) // 12, (p - 1) % 12 + 1


def generar_datos(tipo="patron", n_productos=20, meses=36, semilla=0):
    """Ventas mensuales Poisson alrededor de una media conocida `mu` por producto y mes."""
    rng = np.random.default_rng(semilla)
    hist, inv, mu = [], [], {}
    for i in range(n_productos):
        cn = str(700000 + i)
        base = rng.uniform(20, 120)
        crec = rng.uniform(-0.01, 0.03)          # crecimiento mensual
        amp = rng.uniform(0.2, 0.6)              # amplitud estacional
        fase = int(rng.integers(0, 12))
        for t in range(meses):
            p = PERIODO_INICIO + t
            anio, mes = _anio_mes(p)
            if tipo == "patron":
                media = base * (1 + crec) ** t * (1 + amp * np.sin(2 * np.pi * (mes - 1 - fase) / 12))
            else:
                media = base
            mu[(cn, p)] = media
            hist.append({COL_CN: cn, "Fecha": f"01/{mes:02d}/{anio}", COL_VENTAS: int(rng.poisson(media))})
        inv.append({COL_CN: cn, "Nombre": f"Producto {i}", "Molecula": f"molecula{i % 5}",
                    "Laboratorio": "Lab", "Stock": int(base), "PVL": 3.0})
    return pd.DataFrame(hist), pd.DataFrame(inv), mu


def _contexto(hist, inv):
    return {"df_ventas": hist, "df_inventario": inv, "perfil": PERFIL, "calendario": CALENDARIO, "df_ofertas": None}


def _ultimos_periodos(feat, n):
    per = feat["Anio"] * 12 + feat["Mes"]
    return feat[per > per.max() - n]


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ----------------------------------------------------------------------------- pruebas
def test_sin_fugas_en_variables():
    hist, inv, _ = generar_datos()
    feat = E.build_features(hist, inv, PERFIL, CALENDARIO)
    fechas = pd.to_datetime(hist["Fecha"], dayfirst=True)
    periodo = fechas.dt.year * 12 + fechas.dt.month
    corte = PERIODO_INICIO + 24
    alterado = hist.copy()
    alterado.loc[periodo >= corte, COL_VENTAS] = alterado.loc[periodo >= corte, COL_VENTAS] * 10 + 3
    feat2 = E.build_features(alterado, inv, PERFIL, CALENDARIO)
    claves = [COL_CN, "Anio", "Mes"]
    a = feat[(feat["Anio"] * 12 + feat["Mes"]) <= corte].sort_values(claves).reset_index(drop=True)
    b = feat2[(feat2["Anio"] * 12 + feat2["Mes"]) <= corte].sort_values(claves).reset_index(drop=True)
    distintas = [c for c in E.FEATURE_COLS if not np.allclose(a[c].astype(float), b[c].astype(float))]
    assert not distintas, f"variables que cambian al alterar ventas futuras: {distintas}"
    return f"{len(E.FEATURE_COLS)} variables identicas hasta el mes de corte"


def test_patron_conocido_gana_al_baseline():
    hist, inv, mu = generar_datos("patron")
    feat = E.build_features(hist, inv, PERFIL, CALENDARIO)
    modelo, met, _ = E.entrenar_modelo_ml(feat, contexto=_contexto(hist, inv))
    assert modelo is not None, met.get("error")
    bt = met["backtest"]
    test = _ultimos_periodos(feat, E.MESES_TEST_HOLDOUT)
    oraculo = _rmse(test[COL_VENTAS], [mu[(cn, a * 12 + m)] for cn, a, m in zip(test[COL_CN], test["Anio"], test["Mes"], strict=True)])
    assert bt["mejora_pct"] > 0, f"el ML no gana al baseline con un patron claro ({bt['mejora_pct']}%)"
    assert bt["ml"]["rmse"] >= 0.9 * oraculo, f"error ML {bt['ml']['rmse']} por debajo del minimo posible {oraculo:.2f}: fuga"
    bh = met["backtest_horizonte"]
    return (f"ML {bt['ml']['rmse']} vs baseline {bt['baseline']['rmse']} (mejora {bt['mejora_pct']}%), "
            f"error minimo posible {oraculo:.2f}; a {bh['n_meses']} meses mejora {bh['total']['mejora_pct']}%")


def test_ruido_puro_no_baja_del_minimo():
    hist, inv, mu = generar_datos("ruido", semilla=1)
    feat = E.build_features(hist, inv, PERFIL, CALENDARIO)
    modelo, met, _ = E.entrenar_modelo_ml(feat)
    assert modelo is not None, met.get("error")
    test = _ultimos_periodos(feat, E.MESES_TEST_HOLDOUT)
    oraculo = _rmse(test[COL_VENTAS], [mu[(cn, a * 12 + m)] for cn, a, m in zip(test[COL_CN], test["Anio"], test["Mes"], strict=True)])
    ml = met["backtest"]["ml"]["rmse"]
    assert ml >= 0.9 * oraculo, f"error ML {ml} por debajo del minimo posible {oraculo:.2f}: fuga"
    return f"ML {ml} vs error minimo posible {oraculo:.2f} (baseline {met['backtest']['baseline']['rmse']})"


def test_ventas_barajadas_no_aprende():
    hist, inv, _ = generar_datos("patron", semilla=2)
    rng = np.random.default_rng(2)
    barajado = hist.copy()
    barajado[COL_VENTAS] = rng.permutation(barajado[COL_VENTAS].to_numpy())
    feat = E.build_features(barajado, inv, PERFIL, CALENDARIO)
    modelo, met, _ = E.entrenar_modelo_ml(feat)
    assert modelo is not None, met.get("error")
    test = _ultimos_periodos(feat, E.MESES_TEST_HOLDOUT)
    train = feat[~feat.index.isin(test.index)]
    referencia = _rmse(test[COL_VENTAS], np.full(len(test), train[COL_VENTAS].mean()))
    ml = met["backtest"]["ml"]["rmse"]
    assert ml >= 0.9 * referencia, f"con ventas barajadas el ML acierta ({ml} < {referencia:.2f}): fuga"
    return f"ML {ml} vs predecir la media {referencia:.2f}"


def test_prediccion_futura_cubre_los_meses():
    hist, inv, _ = generar_datos()
    feat = E.build_features(hist, inv, PERFIL, CALENDARIO)
    modelo, _, _ = E.entrenar_modelo_ml(feat)
    ultimo = PERIODO_INICIO + 35                      # dic 2023
    hoy = date(2024, 3, 10)                            # cobertura desde abril 2024
    futuro, sin_datos = E.construir_features_futuras(modelo, hist, inv, PERFIL, CALENDARIO, None, 3, hoy=hoy)
    meses = sorted(set(zip(futuro["Anio"], futuro["Mes"], strict=True)))
    assert meses == [(2024, 4), (2024, 5), (2024, 6)], meses
    assert sin_datos == (2024 * 12 + 4) - 1 - ultimo, sin_datos
    assert futuro.groupby(["Anio", "Mes"])[COL_CN].nunique().eq(len(inv)).all()
    cols = E._columnas_modelo(modelo, futuro)
    assert not futuro[cols].isna().any().any(), "variables con NaN en la prediccion"
    assert (futuro["Prediccion_Base"] >= 0).all()
    return f"meses {meses}, {sin_datos} meses estimados antes de la cobertura"


def test_backtest_horizonte_sin_ver_el_futuro():
    hist, inv, _ = generar_datos()
    feat = E.build_features(hist, inv, PERFIL, CALENDARIO)
    modelo, met, _ = E.entrenar_modelo_ml(feat, contexto=_contexto(hist, inv))
    bh = met["backtest_horizonte"]
    assert bh and len(bh["pasos"]) == E.MESES_TEST_HOLDOUT
    # Repetirlo con las ventas del holdout alteradas no debe cambiar ninguna prediccion.
    fechas = pd.to_datetime(hist["Fecha"], dayfirst=True)
    periodo = fechas.dt.year * 12 + fechas.dt.month
    corte = periodo.max() - E.MESES_TEST_HOLDOUT + 1
    alterado = hist.copy()
    alterado.loc[periodo >= corte, COL_VENTAS] *= 5
    bh2 = E._backtest_horizonte(modelo, _contexto(alterado, inv), corte, periodo.max())
    bh1 = E._backtest_horizonte(modelo, _contexto(hist, inv), corte, periodo.max())
    assert bh1["origen"] == bh2["origen"]
    assert [p["ml"]["sesgo"] for p in bh1["pasos"]] != [p["ml"]["sesgo"] for p in bh2["pasos"]], "no usa ventas reales para medir"
    fut1, _ = E.construir_features_futuras(modelo, hist[periodo < corte], inv, PERFIL, CALENDARIO, None, 3,
                                           hoy=date(*_anio_mes(corte - 1), 15))
    fut2, _ = E.construir_features_futuras(modelo, alterado[periodo < corte], inv, PERFIL, CALENDARIO, None, 3,
                                           hoy=date(*_anio_mes(corte - 1), 15))
    assert np.allclose(fut1["Prediccion_Base"], fut2["Prediccion_Base"]), "la prediccion depende de ventas posteriores al corte"
    return "error por mes: " + ", ".join(f"+{p['h']} ML {p['ml']['rmse']} / base {p['baseline']['rmse']}" for p in bh["pasos"])


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, prueba in pruebas:
        t0 = time.time()
        try:
            detalle = prueba()
            print(f"PASA  {nombre} ({time.time() - t0:.0f}s): {detalle}")
        except AssertionError as e:
            fallos += 1
            print(f"FALLA {nombre}: {e}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas superadas")
    sys.exit(1 if fallos else 0)
