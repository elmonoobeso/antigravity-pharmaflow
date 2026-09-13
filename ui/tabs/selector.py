import streamlit as st
from data.io import crear_farmacia, obtener_farmacias_disponibles


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
