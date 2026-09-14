# PharmaFlow / PharmaSmart

Aplicación de gestión de pedidos e inventario para farmacias, con un motor de
predicción de demanda basado en Machine Learning. Desarrollada por
**Miguel Ángel Leo Acedo** (Supply Chain Analyst con background en Farmacia).

> El repo se creó como `PharmaSmart`; algunos nombres internos (dependencias,
> guía de ML) conservan aún ese nombre. Ver [Notas de nomenclatura](#notas-de-nomenclatura).

## Qué hace

- **Predicción de demanda por producto** con un modelo XGBoost entrenado por
  farmacia (ver [`ml/engine.py`](ml/engine.py)), combinando historial de
  ventas, calendario, clima, grupo terapéutico (ATC) y promociones.
- **Generación de pedidos** por cobertura o por presupuesto, con reglas de
  surtido y tramos de descuento por proveedor.
- **Torre de control / Business Intelligence**: KPIs, roturas de stock,
  productos "zombie" (sin rotación), ROI de campañas.
- **Auditoría de inventario** y generación de informes PDF.
- Motor **heurístico** de respaldo cuando XGBoost/scikit-learn no están
  disponibles, y un modo **ensemble** que combina ambos.

## Stack técnico

- **App:** [Streamlit](https://streamlit.io/)
- **Datos:** pandas, NumPy, Parquet (vía pyarrow), Excel (openpyxl/XlsxWriter)
- **ML:** XGBoost, scikit-learn, joblib
- **Informes:** fpdf2

Ver [`requirements.txt`](requirements.txt) para versiones exactas (verificadas
sobre Python 3.14 en Windows).

## Cómo ejecutar

```bash
py -3.14 -m pip install -r requirements.txt
py -3.14 -m streamlit run app.py
```

Sin `xgboost`/`scikit-learn` instalados, la app sigue funcionando con el
motor heurístico (`ML_AVAILABLE = False` en `ml/engine.py`).

## Estructura del proyecto

```
app.py                  Punto de entrada de la app Streamlit
config/settings.py      Constantes y configuración global
core/
  business.py            Lógica de negocio: pedidos, KPIs, auditoría, benchmarks
  network.py              Lógica de red de farmacias
data/io.py               Carga y persistencia de datos por farmacia
ml/engine.py             Motor de predicción de demanda (XGBoost)
ui/
  charts.py               Gráficos
  components.py           Componentes de UI reutilizables
  tabs/                    Una pestaña por módulo: pedidos, auditoría, BI,
                           configuración, laboratorio ML, torre de control, selector
utils/                   Helpers, red, generación de PDF
farmacias/               Datos y modelos entrenados por farmacia (fixtures de prueba)
tests/test_ml_engine.py  Tests del motor ML
_legacy/                 Código archivado, no se ejecuta (ver LEEME.md)
REVISION.md              Auditoría de bugs y hallazgos por módulo
ml_guide_pharmasmart.md  Guía de conceptos de ML aplicados al proyecto
```

## Motor de Machine Learning

Resumen en [`ml/engine.py`](ml/engine.py):

- Un modelo `XGBRegressor` por farmacia, con objetivo relativo (ventas /
  media de 12 meses) para que el mismo patrón sirva a productos grandes y
  pequeños.
- Split temporal con holdout de los últimos meses, ajuste de hiperparámetros
  por validación temporal, selección automática de grupos de variables
  (backward selection) y backtest walk-forward — nada se mide con datos que
  el modelo pudo ver durante el entrenamiento.
- Se compara siempre contra un baseline honesto (mismo mes del año anterior)
  para verificar que el ML realmente aporta.
- Maneja productos nuevos (cold start) mezclando la predicción con la media
  de su molécula, y calcula stock de seguridad estadístico (Z-score × RMSE
  por producto).

Para una explicación paso a paso de cada concepto usado, ver
[`ml_guide_pharmasmart.md`](ml_guide_pharmasmart.md).

## Estado del proyecto

[`REVISION.md`](REVISION.md) documenta una auditoría en curso del código
(bugs encontrados, severidad y arreglo propuesto por módulo). Antes de tocar
`core/business.py` o `ui/tabs/configuracion.py`, revisa ahí — hay varios
hallazgos ALTO/CRÍTICO aún sin corregir.

## `_legacy/`

Contiene el monolito original (`app (5).py`, 3.777 líneas) y los scripts que
lo dividieron en los módulos actuales. No se ejecuta — se conserva solo como
referencia histórica. Ver [`_legacy/LEEME.md`](_legacy/LEEME.md).

## Notas de nomenclatura

El proyecto nació como "PharmaSmart" y se está renombrando a "PharmaFlow".
Quedan referencias al nombre antiguo en `ml_guide_pharmasmart.md`, el
comentario de cabecera de `requirements.txt` y `pharmasmart.code-workspace`.
