import os

input_file = "app (5).py"
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_line_index(match_str):
    for i, line in enumerate(lines):
        if line.startswith(match_str):
            return i
    return -1

# ui/charts.py extraction
idx_charts_start = get_line_index("def exportar_pedido_excel(df_pedido):")
idx_charts_end = get_line_index("def obtener_ventas_media(meses_cobertura=2):")

if idx_charts_start != -1 and idx_charts_end != -1:
    charts_code = lines[idx_charts_start:idx_charts_end]
    with open("ui/charts.py", "w", encoding="utf-8") as f:
        f.write("import pandas as pd\nimport numpy as np\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom datetime import timedelta, date\nfrom io import BytesIO\nfrom config.settings import COLORS, COL_CN, COL_LAB, COL_NOMBRE, COL_STOCK, COL_PVL, COL_MOLECULA\nfrom utils.helpers import heuristica_tipo_producto\n\n")
        f.writelines(charts_code)
    print("Extracted ui/charts.py")

# core/network.py extraction
idx_net_start = get_line_index("def cargar_red_config():")
idx_net_end = get_line_index("def modulo_selector_farmacia():")

if idx_net_start != -1 and idx_net_end != -1:
    net_code = lines[idx_net_start:idx_net_end]
    with open("core/network.py", "w", encoding="utf-8") as f:
        f.write("import json\nfrom datetime import datetime, date, timedelta\nimport pandas as pd\nimport numpy as np\nfrom config.settings import BASE_DIR, COL_CN, COL_LAB, COL_NOMBRE, COL_PVL, COL_MOLECULA\nfrom data.io import ruta_farmacia_activa\nfrom core.business import aplicar_ofertas_indexado, construir_indice_ofertas, buscar_oferta_por_indice\n\nMARGEN_SEGURIDAD_DIAS = 14\n\n")
        f.writelines(net_code)
    print("Extracted core/network.py")

# ui/tabs.py extraction
idx_tabs_start = idx_net_end
idx_tabs_end = get_line_index("def main():")

if idx_tabs_start != -1 and idx_tabs_end != -1:
    tabs_code = lines[idx_tabs_start:idx_tabs_end]
    with open("ui/tabs.py", "w", encoding="utf-8") as f:
        f.write("import streamlit as st\nimport pandas as pd\nimport numpy as np\nfrom datetime import date, datetime, timedelta\nfrom config.settings import COL_CN, COL_LAB, COL_NOMBRE, COL_STOCK, COL_PVL, COL_MOLECULA, COL_FECHA, COL_VENTAS, HEALTH_SCORE_MESES_DEFAULT, COLORS\n")
        f.write("from data.io import obtener_farmacias_disponibles, crear_farmacia, ruta_farmacia_activa, guardar_dataframe_farmacia, cargar_json_farmacia\n")
        f.write("from utils.helpers import validar_y_renombrar_columnas, safe_div, format_eur\n")
        f.write("from core.business import *\n")
        f.write("from core.network import *\n")
        f.write("from ml.engine import ML_AVAILABLE, entrenar_modelo_ml, cargar_modelo_farmacia, guardar_modelo_farmacia, build_features, cold_start_proxy, predecir_demanda_ensemble\n")
        f.write("from ui.components import render_header, render_kpi\n")
        f.write("from ui.charts import *\n")
        f.write("from utils.pdf_generator import generar_informe_pdf\n\n")
        f.writelines(tabs_code)
    print("Extracted ui/tabs.py")
