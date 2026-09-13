import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from config.settings import COL_CN, COL_LAB, COL_NOMBRE, COL_STOCK, COL_PVL, COL_MOLECULA, COL_FECHA, COL_VENTAS, HEALTH_SCORE_MESES_DEFAULT, COLORS
from data.io import obtener_farmacias_disponibles, crear_farmacia, ruta_farmacia_activa, guardar_dataframe_farmacia, cargar_json_farmacia
from utils.helpers import validar_y_renombrar_columnas, safe_div, format_eur
from core.business import *
from core.network import *
from ml.engine import (ML_AVAILABLE, entrenar_modelo_ml, guardar_modelo_farmacia,
                       build_features, cold_start_proxy, obtener_modelo_cacheado,
                       invalidar_cache_modelo, necesita_reentrenamiento,
                       generar_pedido_ml, generar_pedido_ensemble)
from ui.components import render_kpi
from ui.charts import *
from utils.pdf_generator import generar_informe_pdf
from utils.network import ejecutar_benchmark_red

def modulo_selector_farmacia():
    st.markdown("### \U0001f3e5 Selecciona la Farmacia")
    farmacias = obtener_farmacias_disponibles()
    col_sel, col_new = st.columns([2, 1])
    with col_sel:
        if farmacias:
            seleccion = st.selectbox("Farmacias disponibles:", ["-- Selecciona --"] + farmacias, key="sel_farmacia")
            if seleccion != "-- Selecciona --":
                if st.button("\u2705 Abrir Farmacia", type="primary", width='stretch'):
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
        if st.button("\U0001f4be Crear", width='stretch') and nn.strip():
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
        st.dataframe(df_of_raw.head(5), width='stretch', hide_index=True)
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
    df_reg_e = st.data_editor(df_reg, num_rows="dynamic", width='stretch', key="ed_reg")
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
    df_pr_e = st.data_editor(df_pr, num_rows="dynamic", width='stretch', key="ed_prot")
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
            if st.button("\U0001f9e0 Entrenar Modelo ML", type="primary", width='stretch'):
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
                        if fig_imp: st.plotly_chart(fig_imp, width='stretch')
                    else:
                        st.error(f"Error: {metricas.get('error','Desconocido')}")

    # --- I. Benchmark de Red ---
    st.divider()
    st.markdown("#### I. Benchmark de Red")
    st.caption("Verifica la latencia de tu conexión frente a endpoints globales.")
    if st.button("🌐 Iniciar Prueba de Red"):
        with st.spinner("Midiendo latencia..."):
            res = ejecutar_benchmark_red()
        
        if res["sana"]:
            st.success("✅ Conexión de red saludable.")
        else:
            st.warning("⚠️ Posibles problemas de red o firewall bloqueando el tráfico.")
            
        c_bench = st.columns(len(res["resultados"]))
        for i, r in enumerate(res["resultados"]):
            with c_bench[i]:
                lat = f"{r['latencia_ms']} ms" if r['latencia_ms'] is not None else "Timeout/Error"
                render_kpi(r['url'].replace("https://", ""), lat, "OK" if r['estado'] == "OK" else r['estado'], r['estado'] == "OK")

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
        st.plotly_chart(grafico_gauge_health(hs), width='stretch', config={"displayModeBar":False})
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
            st.plotly_chart(fig_cal, width='stretch', config={"displayModeBar":False})
        else:
            st.info("Sin datos de ventas para calcular calendario de reposicion.")

    # === 3. Dinero en Riesgo + Productos con peor rotacion ===
    st.markdown("---"); st.markdown("#### \U0001f4b0 Dinero en Riesgo")
    c_dr, c_top = st.columns([3, 2])
    with c_dr:
        fig_dr = grafico_dinero_en_riesgo_donut(dz, 0, coste_op)
        if fig_dr:
            st.plotly_chart(fig_dr, width='stretch', config={"displayModeBar":False})
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
                st.dataframe(df_peor[cols_show], hide_index=True, width='stretch')
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
            st.plotly_chart(fig_wf, width='stretch', config={"displayModeBar":False})
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
            if fh: st.plotly_chart(fh, width='stretch', config={"displayModeBar":False})
        with c2:
            if fzr: st.plotly_chart(fzr, width='stretch', config={"displayModeBar":False})
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
            st.plotly_chart(fig_roi, width='stretch', config={"displayModeBar":False})
        with st.expander("Ver tabla completa de ROI"):
            df_roi_show = df_roi.copy()
            df_roi_show["Stock_EUR"] = df_roi_show["Stock_EUR"].apply(lambda x: f"{x:,.2f} \u20ac")
            df_roi_show["Venta_Anual_EUR"] = df_roi_show["Venta_Anual_EUR"].apply(lambda x: f"{x:,.2f} \u20ac")
            df_roi_show["ROI"] = df_roi_show["ROI"].apply(lambda x: f"{x:.2f}x")
            st.dataframe(df_roi_show, width='stretch', hide_index=True)

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
                st.dataframe(df_rot_show[cols_show].head(30), width='stretch', hide_index=True)
        else:
            st.info("Sin datos de rotacion.")
    with c_co:
        st.markdown("**\U0001f4c9 Coste de Oportunidad de Roturas**")
        if coste_op > 0:
            st.metric("Venta perdida estimada", f"{coste_op:,.2f} \u20ac/mes")
            if not df_coste_op.empty:
                with st.expander(f"Ver {len(df_coste_op)} productos en rotura"):
                    st.dataframe(df_coste_op.head(20), width='stretch', hide_index=True)
        else:
            st.success("\u2705 Sin roturas — no hay venta perdida")

    # === 8. Zombie + UVI ===
    st.markdown("---"); st.markdown("#### \U0001f9df Stock Inmovilizado")
    c_z, c_u = st.columns(2)
    with c_z:
        st.markdown(f"**Zombie (12m sin ventas)** — {len(df_z)} prods | {format_eur(dz)}")
        if not df_z.empty:
            cs = [c for c in [COL_CN,COL_NOMBRE,COL_LAB,COL_STOCK,COL_PVL,"Valor_Inmovilizado"] if c in df_z.columns]
            st.dataframe(df_z[cs].sort_values("Valor_Inmovilizado",ascending=False).head(20), width='stretch', hide_index=True)
        else:
            st.success("\u2705 Sin stock zombie")
    with c_u:
        st.markdown(f"**UVI (6m sin ventas)** — {len(df_uvi)} prods | {format_eur(duvi)}")
        if not df_uvi.empty:
            cs = [c for c in [COL_CN,COL_NOMBRE,COL_LAB,COL_STOCK,COL_PVL,"Valor_Inmovilizado"] if c in df_uvi.columns]
            st.dataframe(df_uvi[cs].sort_values("Valor_Inmovilizado",ascending=False).head(20), width='stretch', hide_index=True)
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
        if fig_imp: st.plotly_chart(fig_imp, width='stretch', config={"displayModeBar":False})

    # === 10. Descargar Informe ===
    st.markdown("---")
    farmacia_nombre = st.session_state.get("farmacia_activa", "Farmacia").replace("_", " ").title()
    ahorro_acum_total = df_hk["ahorro_pedido"].sum() if not df_hk.empty else 0
    informe_bytes = generar_informe_pdf(
        farmacia_nombre, hs, len(df_z), dz, len(df_uvi), duvi,
        nr, coste_op, ahorro_acum_total, rotacion_media, benchmark)
    
    informe_filename = "informe_pharmasmart.pdf"
    
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
    if st.button("\u26a1 Generar Pedido", type="primary", width='stretch'):
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
                st.dataframe(dd, width='stretch', hide_index=True)

        st.markdown("---")
        lab_txt = lab_sel if lab_sel != "-- Todos --" else "Todos"
        col_dl, col_confirm = st.columns(2)
        with col_dl:
            st.download_button(
                f"\U0001f4e5 Descargar Pedido {lab_txt} (.xlsx)",
                data=exportar_pedido_excel(df_ped),
                file_name=f"PharmaSmart_{lab_txt}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary", width='stretch')
        with col_confirm:
            if st.button("\u2705 Confirmar Pedido", width='stretch', type="secondary"):
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
                st.dataframe(df_surt_filtered.rename(columns={"Presentacion/Molecula": "Regla / Molécula"}), width='stretch', hide_index=True)
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

    c_op, _ = calcular_coste_oportunidad(df_inv, df_vm)
    
    # Nuevas variables
    fill_rate, total_con_demanda, n_roturas_hab = calcular_indice_servicio(df_inv, df_hist)
    val_cad, prods_cad, cad_real = calcular_riesgo_caducidad(df_inv)
    
    
    # --- RESUMEN EJECUTIVO (Scorecard) ---
    st.markdown("#### \U0001f4cb Resumen Ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(grafico_gauge_health(hs), width='stretch', config={"displayModeBar":False}, key="gauge_auditoria")
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
            st.plotly_chart(fig_donut, width='stretch', config={"displayModeBar":False}, key="donut_auditoria")
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
            if fig_tail: st.plotly_chart(fig_tail, width='stretch', config={"displayModeBar":False}, key="tail_auditoria")
            
        with c_r:
            st.markdown("**Inversión por Laboratorio**")
            tipo_filtro = st.radio("Filtro:", ["Todos", "Medicamento", "Parafarmacia"], horizontal=True)
            fig_labs = grafico_distribucion_laboratorios(df_inv, tipo_filtro)
            if fig_labs: st.plotly_chart(fig_labs, width='stretch', config={"displayModeBar":False}, key="labs_auditoria")
            
        st.markdown("#### Análisis ABC (Valor de Inmovilizado)")
        df_abc = calcular_analisis_abc(df_inv, df_vm)
        if not df_abc.empty:
            df_abc_show = df_abc[[COL_NOMBRE, COL_LAB, COL_STOCK, "Valor_Stock", "Clasificacion_ABC", "Venta_Media_Mensual"]].head(25)
            st.dataframe(df_abc_show, width='stretch', hide_index=True)

    with ta2:
        st.markdown("#### \u26a0\ufe0f Riesgo Operacional (Inmovilizado)")
        c_z, c_u = st.columns(2)
        with c_z:
            st.markdown(f"**\U0001f9df Stock Zombie (12m sin ventas)** — {format_eur(dz)}")
            if not df_z.empty:
                st.dataframe(df_z[[COL_NOMBRE, COL_LAB, COL_STOCK, "Valor_Inmovilizado"]].sort_values("Valor_Inmovilizado", ascending=False).head(15), width='stretch', hide_index=True)
            else: st.success("Sin zombies")
        with c_u:
            st.markdown(f"**\U0001fa79 Stock UVI (6m sin ventas)** — {format_eur(du)}")
            if not df_u.empty:
                st.dataframe(df_u[[COL_NOMBRE, COL_LAB, COL_STOCK, "Valor_Inmovilizado"]].sort_values("Valor_Inmovilizado", ascending=False).head(15), width='stretch', hide_index=True)
            else: st.success("Sin UVI")
            
        st.markdown("---")
        c_hc, c_cr = st.columns([1.5, 1])
        with c_hc:
            fig_heat = grafico_heatmap_cobertura(df_inv, df_vm)
            if fig_heat: st.plotly_chart(fig_heat, width='stretch', config={"displayModeBar":False}, key="heat_auditoria")
        with c_cr:
            df_exceso = df_m[df_m["Exceso"] > 0]
            fig_conc = grafico_concentracion_riesgo(df_z, df_u, df_exceso)
            if fig_conc: st.plotly_chart(fig_conc, width='stretch', config={"displayModeBar":False}, key="conc_auditoria")
            
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
                st.dataframe(res, width='stretch', hide_index=True)
            else:
                st.info("Se requiere columna PVP para calcular matriz real.")
                
        with c_est:
            st.markdown("**Dependencia Estacional (Riesgo de Liquidez)**")
            df_est = calcular_dependencia_estacional(df_hist)
            if not df_est.empty:
                fig_est = grafico_estacionalidad_liquidez(df_est)
                if fig_est: st.plotly_chart(fig_est, width='stretch', config={"displayModeBar":False}, key="est_auditoria")
                
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
                    st.dataframe(df_muest[[COL_NOMBRE, "Stock_Teorico", "Stock_Fisico", "Descuadre_Uds", "Descuadre_Eur"]].sort_values("Descuadre_Eur"), width='stretch', hide_index=True)
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
        st.plotly_chart(fig_gantt, width='stretch', config={"displayModeBar": False})

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
                        st.dataframe(df_comp, width='stretch', hide_index=True)

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
                            if st.button("\u2705 Marcar Ejecutada", key=f"exec_{idx}"):
                                registrar_compra_conjunta(
                                    farms, lab_filtro, sim["ahorro_total"], sim["resumen_farmacias"])
                                st.success("\u2705 Compra conjunta registrada.")
                                st.rerun()
                        with col_lost:
                            if st.button("\u274c No Ejecutada", key=f"lost_{idx}"):
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
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    else:
        st.info("No hay compras conjuntas registradas aun.")


# ===========================================================================
# MAIN
# ===========================================================================
