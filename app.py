import streamlit as st
from datetime import datetime
from config.settings import APP_NAME, APP_ICON, VERSION, COLORS
from utils.helpers import inject_custom_css
from ui.components import render_header
from data.io import _autocargar_datos_farmacia
from core.network import generar_notificaciones
from ui.tabs import (
    modulo_selector_farmacia,
    modulo_configuracion,
    modulo_business_intelligence,
    modulo_generador_pedidos,
    modulo_auditoria,
    modulo_torre_control
)

def main():
    st.set_page_config(page_title=f"{APP_NAME} \u2014 Gestion de Compras",
        page_icon=APP_ICON, layout="wide", initial_sidebar_state="collapsed")
    inject_custom_css()

    if "farmacia_activa" not in st.session_state:
        render_header()
        modulo_selector_farmacia()
        return

    farmacia = st.session_state["farmacia_activa"]
    
    # Auto-cargar datos desde disco si no estan en session_state
    datos_cargados = _autocargar_datos_farmacia()
    if datos_cargados:
        st.toast(f"\U0001f4c2 Datos restaurados: {', '.join(datos_cargados)}", icon="\u2705")
        
    # Notificaciones (campanita) \u2014 cacheado en session_state para evitar lecturas redundantes
    if "alertas_red_cache" not in st.session_state:
        st.session_state["alertas_red_cache"] = generar_notificaciones()
        
    alertas = st.session_state["alertas_red_cache"]
    n_alertas = len(alertas)
    bell = f" \U0001f514 {n_alertas}" if n_alertas > 0 else ""
    render_header(farmacia_nombre=farmacia.replace("_", " ").title() + bell)

    with st.sidebar:
        st.markdown(f"**Farmacia:** {farmacia.replace('_', ' ').title()}")
        if n_alertas > 0:
            st.markdown(f"\U0001f514 **{n_alertas} oportunidades** de compra conjunta")
            for al in alertas[:3]:
                farms_txt = " + ".join(f.replace("_", " ").title() for f in al["farmacias"])
                st.caption(f"\u2022 {al['laboratorio']}: {farms_txt}")
        if st.button("\U0001f504 Cambiar Farmacia"):
            for k in list(st.session_state.keys()):
                if k != "farmacia_activa":
                    st.session_state.pop(k, None)
            st.session_state.pop("farmacia_activa", None)
            st.rerun()

    t1, t2, t3, t4, t5 = st.tabs([
        "\u2699\ufe0f Configuracion",
        "\U0001f4ca Business Intelligence",
        "\U0001f680 Pedidos Transfer",
        "\U0001f6e0\ufe0f Auditoria",
        "\U0001f3d7\ufe0f Torre de Control",
    ])
    
    with t1: modulo_configuracion()
    with t2: modulo_business_intelligence()
    with t3: modulo_generador_pedidos()
    with t4: modulo_auditoria()
    with t5: modulo_torre_control()

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:{COLORS['muted']};font-size:0.8rem;'>"
        f"{APP_ICON} {APP_NAME} v{VERSION} \u00b7 {farmacia.replace('_', ' ').title()} \u00b7 "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
