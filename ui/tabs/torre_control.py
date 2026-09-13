import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from config.settings import COLORS, COL_CN, COL_MOLECULA, COL_NOMBRE
from core.network import cargar_ahorro_perdido, cargar_historico_compras_conjuntas, cargar_pedidos_confirmados, cargar_red_config, guardar_red_config, obtener_ventanas_red, optimizar_timing_red, registrar_ahorro_perdido, registrar_compra_conjunta, simular_pedido_conjunto
from data.io import obtener_farmacias_disponibles
from ui.components import render_kpi
from utils.helpers import format_eur, safe_div


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
