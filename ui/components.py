import streamlit as st
from config.settings import APP_ICON, APP_NAME, VERSION

def render_header(farmacia_nombre=""):
    subtitulo = f"Farmacia: {farmacia_nombre}" if farmacia_nombre else "Sistema Inteligente de Gestion de Compras"
    st.markdown(f"""
    <div class="pharma-header">
        <div><h1>{APP_ICON} {APP_NAME}</h1>
        <span style="opacity:0.85;font-size:0.9rem;">{subtitulo}</span></div>
        <span class="version">v{VERSION}</span>
    </div>""", unsafe_allow_html=True)

def render_kpi(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_positive else "negative"
        icon = "\u2191" if delta_positive else "\u2193"
        delta_html = f'<div class="kpi-delta {cls}">{icon} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)
