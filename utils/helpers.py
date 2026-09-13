import streamlit as st
import pandas as pd

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .pharma-header {
        background: linear-gradient(135deg, #0066FF 0%, #00C49A 100%);
        padding: 1.5rem 2rem; border-radius: 16px; color: white;
        margin-bottom: 1.5rem; display: flex; align-items: center;
        justify-content: space-between;
    }
    .pharma-header h1 { margin:0; font-size:1.8rem; font-weight:700; }
    .pharma-header .version {
        background: rgba(255,255,255,0.2); padding: 4px 12px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
    .kpi-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 1.25rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .kpi-label { font-size:0.8rem; color:#64748B; font-weight:500;
        text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }
    .kpi-value { font-size:1.8rem; font-weight:700; color:#1E293B; line-height:1.2; }
    .kpi-delta { font-size:0.85rem; font-weight:500; margin-top:4px; }
    .kpi-delta.positive { color:#10B981; }
    .kpi-delta.negative { color:#FF4B4B; }
    .upload-zone { border:2px dashed #CBD5E1; border-radius:12px; padding:2rem;
        text-align:center; background:#F8FAFC; transition:border-color 0.2s; }
    .upload-zone:hover { border-color:#0066FF; }
    .budget-bar-container { background:#E2E8F0; border-radius:8px;
        overflow:hidden; height:24px; position:relative; }
    .budget-bar-fill { height:100%; border-radius:8px; transition:width 0.4s ease;
        display:flex; align-items:center; justify-content:center;
        font-size:0.75rem; font-weight:600; color:white; }
    .dataframe { font-size: 0.85rem !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0;
        padding:10px 20px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

def format_eur(value):
    if abs(value) >= 1000:
        return f"{value:,.2f} \u20ac".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:.2f} \u20ac".replace(".", ",")

def safe_div(a, b, default=0):
    """Division segura evitando ZeroDivisionError."""
    try:
        return float(a) / float(b) if round(float(b), 4) != 0 else default
    except (ValueError, TypeError):
        return default

def validar_y_renombrar_columnas(df, columnas_requeridas):
    df = df.copy()
    informe = {}
    rename_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for col_esperada in columnas_requeridas:
        col_lower = col_esperada.lower().strip()
        encontrada = None
        if col_lower in cols_lower:
            encontrada = cols_lower[col_lower]
        else:
            for df_col_lower, df_col_original in cols_lower.items():
                if col_lower in df_col_lower or df_col_lower in col_lower:
                    encontrada = df_col_original
                    break
        if encontrada and encontrada != col_esperada:
            rename_map[encontrada] = col_esperada
        informe[col_esperada] = encontrada
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, informe
