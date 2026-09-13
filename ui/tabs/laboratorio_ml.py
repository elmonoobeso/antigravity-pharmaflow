from ml.engine import ML_AVAILABLE
import streamlit as st
import pandas as pd
from data.io import cargar_json_farmacia
from ui.charts import grafico_backtest_folds, grafico_backtest_totales, grafico_importancia_features, grafico_residuos, grafico_seleccion_variables, grafico_timeline_split, grafico_tuning, grafico_walk_forward
from ui.components import render_kpi


def modulo_laboratorio_ml():
    st.markdown("### \U0001f9ea Laboratorio ML")
    st.caption("Cada fase del proceso, con los numeros que la respaldan. "
               "Lo que se mide aqui nunca se uso para entrenar ni para elegir la configuracion.")

    if not ML_AVAILABLE:
        st.warning("Instala xgboost y scikit-learn para usar el motor ML.")
        return

    pipeline = cargar_json_farmacia("modelo_pipeline.json", default={})
    if not pipeline:
        st.info("Todavia no hay un entrenamiento registrado. "
                "Ve a **Configuracion -> Motor de Prediccion ML** y entrena el modelo.")
        return

    bt = pipeline.get("backtest", {})
    ml_g, base_g = bt.get("ml", {}), bt.get("baseline", {})
    mejora = bt.get("mejora_pct")

    # --- 0. Veredicto ---
    st.markdown("#### Veredicto")
    if mejora is None:
        st.info("Sin backtest suficiente para emitir un veredicto.")
    elif mejora > 5:
        st.success(f"**El ML aporta valor**: reduce el error un {mejora}% frente al metodo "
                   "heuristico simple. Merece la pena usar el motor ML o Ensemble.")
    elif mejora > 0:
        st.warning(f"**Mejora marginal** ({mejora}%). El ML apenas supera al heuristico: "
                   "la diferencia puede no justificar su complejidad en esta farmacia.")
    else:
        st.error(f"**El ML no supera al baseline** ({mejora}%). Para esta farmacia conviene "
                 "usar el motor Heuristico. Un modelo que no le gana a repetir el mismo mes "
                 "del ano pasado no deberia decidir los pedidos.")

    k1, k2, k3, k4 = st.columns(4)
    with k1: render_kpi("RMSE ML", str(ml_g.get("rmse", "?")))
    with k2: render_kpi("RMSE Baseline", str(base_g.get("rmse", "?")))
    with k3: render_kpi("Mejora", f"{mejora}%" if mejora is not None else "?",
                        delta_positive=(mejora or 0) > 0)
    with k4: render_kpi("MAPE ML", f"{ml_g.get('mape', '?')}%")
    st.caption(f"Entrenado el {pipeline.get('fecha', '?')}")

    # --- 1. Datos ---
    st.divider()
    st.markdown("#### 1. Datos de partida")
    d = pipeline.get("datos", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi("Filas", f"{d.get('n_filas', 0):,}".replace(",", "."))
    with c2: render_kpi("Productos", str(d.get("n_productos", 0)))
    with c3: render_kpi("Meses", str(d.get("n_periodos", 0)))
    with c4: render_kpi("Rango", f"{d.get('desde', '?')} - {d.get('hasta', '?')}")
    st.caption("Una fila = un producto en un mes. El historico se agrega a nivel mensual "
               "porque es la unidad en la que se decide un pedido.")

    # --- 2. Features ---
    st.divider()
    st.markdown("#### 2. Variables que usa el modelo")
    feats = pipeline.get("features", {})
    sel = feats.get("seleccion", {})
    st.caption(f"{len(feats.get('usadas', []))} variables en el modelo final. La seleccion es automatica y por "
               "grupos: se reentrena quitando cada grupo y solo se queda si quitarlo empeora el error. Se decide "
               "con meses de entrenamiento; el holdout no participa.")
    cs1, cs2 = st.columns(2)
    with cs1:
        fig_sel = grafico_seleccion_variables(sel)
        if fig_sel:
            st.plotly_chart(fig_sel, width="stretch", key="lab_seleccion")
    with cs2:
        fig_imp = grafico_importancia_features({"importance": feats.get("importancia", {})})
        if fig_imp:
            st.plotly_chart(fig_imp, width="stretch", key="lab_importancia")
    if sel.get("grupos"):
        st.dataframe(pd.DataFrame([{
            "Grupo": g["grupo"], "Variables": ", ".join(g["variables"]),
            "Error sin el grupo": g["rmse_sin_grupo"], "Impacto al quitarlo (%)": g["impacto_pct"],
            "Decision": g["decision"],
        } for g in sel["grupos"]]), width="stretch", hide_index=True)
        st.caption(f"RMSE de validacion con todas las variables: {sel.get('rmse_todas')} · "
                   f"con las elegidas: {sel.get('rmse_elegidas')}")
        if sel.get("nota"):
            st.info(sel["nota"])

    # --- 3. Split temporal ---
    st.divider()
    st.markdown("#### 3. Separacion temporal")
    sp = pipeline.get("split", {})
    st.caption(f"Los ultimos {sp.get('meses_holdout', 3)} meses (desde {sp.get('periodo_corte', '?')}) "
               "se apartan antes de tocar nada. No entrenan ni eligen hiperparametros: solo miden. "
               "Un corte aleatorio mezclaria futuro y pasado, e inflaria el resultado.")
    fig_split = grafico_timeline_split(pipeline)
    if fig_split:
        st.plotly_chart(fig_split, width="stretch", key="lab_split")

    # --- 4. Tuning ---
    st.divider()
    st.markdown("#### 4. Eleccion de hiperparametros")
    tun = pipeline.get("tuning", {})
    st.caption(f"Metodo: {tun.get('metodo', '?')}. Se prueban combinaciones sobre cortes "
               "temporales dentro del periodo de entrenamiento. El holdout no participa: si la "
               "configuracion se eligiera mirandolo, dejaria de ser una medida honesta.")
    fig_tun = grafico_tuning(tun.get("ensayos", []), tun.get("elegidos", {}))
    if fig_tun:
        st.plotly_chart(fig_tun, width="stretch", key="lab_tuning")
    st.info(f"Configuracion elegida: {tun.get('elegidos', {})}")

    # --- 5. Backtest ---
    st.divider()
    st.markdown("#### 5. Backtest walk-forward")
    st.caption("Se reentrena mes a mes y se predice el mes siguiente, replicando lo que pasaria "
               "en uso real. Es la medida en la que mas se puede confiar.")
    folds = bt.get("folds", [])
    fig_wf = grafico_walk_forward(pipeline)
    if fig_wf:
        st.plotly_chart(fig_wf, width="stretch", key="lab_walk_forward")
    if folds:
        df_folds = pd.DataFrame([{
            "Mes predicho": f["periodo"],
            "Filas entrenamiento": f["n_train"],
            "RMSE ML": f["ml"]["rmse"],
            "RMSE Baseline": f["baseline"]["rmse"],
            "MAPE ML": f["ml"]["mape"],
            "Sesgo ML": f["ml"]["sesgo"],
        } for f in folds])
        st.dataframe(df_folds, width="stretch", hide_index=True)
        cg1, cg2 = st.columns(2)
        with cg1:
            fig_bt = grafico_backtest_folds(bt)
            if fig_bt: st.plotly_chart(fig_bt, width="stretch", key="lab_folds")
        with cg2:
            fig_tot = grafico_backtest_totales(bt)
            if fig_tot: st.plotly_chart(fig_tot, width="stretch", key="lab_totales")

    # --- 6. Holdout vs baseline ---
    st.divider()
    st.markdown("#### 6. Holdout final frente al baseline")
    ho, hb = pipeline.get("holdout", {}), pipeline.get("holdout_baseline", {})
    if ho:
        df_cmp = pd.DataFrame([
            {"Metrica": "RMSE (error tipico)", "ML": ho.get("rmse"), "Baseline": hb.get("rmse")},
            {"Metrica": "MAE (error medio)", "ML": ho.get("mae"), "Baseline": hb.get("mae")},
            {"Metrica": "MAPE (% de error)", "ML": ho.get("mape"), "Baseline": hb.get("mape")},
            {"Metrica": "Sesgo (+ pasa / - se queda corto)", "ML": ho.get("sesgo"), "Baseline": hb.get("sesgo")},
        ])
        st.dataframe(df_cmp, width="stretch", hide_index=True)
        st.caption(f"R2 del holdout: {ho.get('r2', '?')}. Ojo: el R2 compara contra predecir "
                   "siempre la media, que en ventas regulares es un liston muy bajo. Por eso la "
                   "referencia util es el baseline, no el R2.")

    # --- 7. Analisis de error ---
    st.divider()
    st.markdown("#### 7. Donde falla")
    ce1, ce2 = st.columns([3, 2])
    with ce1:
        fig_res = grafico_residuos(pipeline.get("residuos_hist", {}))
        if fig_res:
            st.plotly_chart(fig_res, width="stretch", key="lab_residuos")
            h = pipeline.get("residuos_hist", {})
            st.caption(f"Error mediano: {h.get('p50', '?')} uds. El 90% de las predicciones "
                       f"falla en menos de {h.get('p90', '?')} unidades.")
    with ce2:
        peores = pipeline.get("peores_productos", [])
        if peores:
            st.markdown("**Productos con mas error**")
            st.dataframe(pd.DataFrame(peores).rename(columns={"_err": "Error medio (uds)"}),
                         width="stretch", hide_index=True)

    # --- 8. Traduccion a negocio ---
    st.divider()
    st.markdown("#### 8. Como se traduce en el pedido")
    rmse_ref = ml_g.get("rmse", 0) or 0
    st.markdown(
        "El error del backtest **no se queda en una metrica**: es lo que fija el colchon de seguridad.\n\n"
        "- **Stock de seguridad** = z x RMSE del producto x raiz(meses de cobertura), con z = 1,645 al 95% de nivel de "
        "servicio y z = 2,326 al 99%.\n"
        f"- Con el RMSE global actual ({rmse_ref} uds) y 1 mes de cobertura, el colchon seria de "
        f"**{round(1.645 * rmse_ref)} uds al 95%** o **{round(2.326 * rmse_ref)} uds al 99%**.\n"
        "- El RMSE se calcula **por producto**: los predecibles reciben menos colchon y los "
        "erraticos mas, en vez de aplicar un margen plano a todo el inventario.\n"
        "- En el motor **Ensemble**, ese mismo error decide cuanto pesa el ML frente al "
        "heuristico: si el modelo falla mas que la propia variabilidad del producto, pesa menos."
    )
    st.caption("Por eso importa que el RMSE venga del backtest y no de un corte aleatorio: "
               "un error subestimado se traduce en menos colchon del necesario y en roturas.")
