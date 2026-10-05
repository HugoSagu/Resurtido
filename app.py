from datetime import datetime
from pathlib import Path
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import ReplenishmentEngine

# ----------------- CONFIGURACIÓN DE PÁGINA -----------------
st.set_page_config(
    page_title="UPPER | Pedido sugerido - Proveedores directos",
    page_icon="🚩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- DETECCIÓN DE ARCHIVOS E IMÁGENES -----------------
BASE_DIR = Path(__file__).resolve().parent

# Buscar logo PNG
logo_files = list(BASE_DIR.glob("*.png"))
LOGO_PATH = BASE_DIR / "logo.png" if (BASE_DIR / "logo.png").exists() else (logo_files[0] if logo_files else None)

# Buscar fondo WEBP
bg_files = list(BASE_DIR.glob("*.webp"))
BG_PATH = BASE_DIR / "fondo.webp" if (BASE_DIR / "fondo.webp").exists() else (bg_files[0] if bg_files else None)

def get_base64_encoded_image(image_path: Path) -> str:
    """Codifica una imagen local a base64 para inyectar en CSS de fondo."""
    if image_path and image_path.exists():
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return ""

bg_base64 = get_base64_encoded_image(BG_PATH)

# Estilo de fondo condicional
bg_css_rule = f"""
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.78), rgba(255, 255, 255, 0.78)), 
                    url("data:image/webp;base64,{bg_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
""" if bg_base64 else """
    .stApp {
        background-color: #F8F9FA;
    }
"""

# ----------------- ESTILOS UPPER ENERGY (CSS) -----------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Montserrat', sans-serif;
    }}

    {bg_css_rule}

    :root {{
        --upper-red: #E51A24;
        --upper-red-hover: #C4121B;
        --upper-dark: #121212;
        --upper-card-border: #E5E7EB;
    }}

    /* Header Superior */
    .upper-header {{
        background: linear-gradient(90deg, #111111 0%, #1F1F1F 70%, #E51A24 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .upper-header h1 {{
        font-weight: 900;
        letter-spacing: 1.5px;
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2.1rem;
    }}
    .upper-header span {{
        color: #E51A24;
    }}
    .upper-header p {{
        color: #CCCCCC;
        margin-top: 5px;
        font-size: 0.9rem;
        margin-bottom: 0;
    }}

    /* Tarjetas de Métricas */
    div[data-testid="stMetric"] {{
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid var(--upper-card-border);
        border-left: 5px solid var(--upper-red);
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
    }}
    div[data-testid="stMetricLabel"] p {{
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        color: #444444 !important;
    }}
    div[data-testid="stMetricValue"] div {{
        color: #111111 !important;
        font-weight: 800 !important;
    }}

    /* Botón de Descarga */
    .stDownloadButton > button {{
        background-color: var(--upper-red) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 12px rgba(229, 26, 36, 0.3) !important;
        transition: all 0.2s ease !important;
    }}
    .stDownloadButton > button:hover {{
        background-color: var(--upper-red-hover) !important;
        transform: translateY(-1px);
    }}

    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid #E5E7EB;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 700;
        font-size: 1rem;
        padding: 10px 20px;
        color: #555555;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--upper-red) !important;
        border-bottom: 3px solid var(--upper-red) !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: rgba(250, 250, 250, 0.95);
        border-right: 1px solid #E5E7EB;
    }}
</style>
""", unsafe_allow_html=True)


# ----------------- MOTOR Y BASES DE DATOS -----------------
engine = ReplenishmentEngine(lead_time_days=3, review_frequency_days=7, extra_coverage_days=2, min_cv=0.50)

VENTAS_PATH = BASE_DIR / "ventas.parquet"
CATALOGO_PATH = BASE_DIR / "catalogo.parquet"
EXISTENCIAS_PATH = BASE_DIR / "existencias.parquet"
ESTACIONALIDAD_PATH = BASE_DIR / "estacionalidad.parquet"

ARCHIVOS_REQUERIDOS = [VENTAS_PATH, CATALOGO_PATH, EXISTENCIAS_PATH, ESTACIONALIDAD_PATH]
if not all(p.is_file() for p in ARCHIVOS_REQUERIDOS):
    st.error("🚨 Faltan bases de datos Parquet en la carpeta del proyecto.")
    st.stop()


# ----------------- PIPELINE VECTORIZADO -----------------
@st.cache_data(ttl=None, show_spinner="Calculando reabastecimiento Upper...")
def load_and_compute_pipeline(mes_objetivo: int) -> pd.DataFrame:
    df_cat = pd.read_parquet(CATALOGO_PATH)
    df_cat["Tipo de proveedor"] = df_cat["Tipo de proveedor"].astype(str).str.strip().str.upper()
    df_cat_dsd = df_cat[df_cat["Tipo de proveedor"] == "DIRECTO"].copy()

    if df_cat_dsd.empty:
        return pd.DataFrame()

    df_exist = pd.read_parquet(EXISTENCIAS_PATH)
    cols_exist = [c for c in ["Establecimiento", "Código Mercancía", "Existencia", "Descripción"] if c in df_exist.columns]
    df_exist = df_exist[cols_exist].copy()

    df_estac = pd.read_parquet(ESTACIONALIDAD_PATH)
    df_estac_mes = df_estac[df_estac["Mes"] == mes_objetivo][["Nivel 2", "Factor_Estacional"]].copy()

    df_cat_estac = pd.merge(df_cat_dsd, df_estac_mes, on="Nivel 2", how="left")
    df_cat_estac["Factor_Estacional"] = df_cat_estac["Factor_Estacional"].fillna(1.0).astype("float32")

    master_base = pd.merge(df_cat_estac, df_exist, on="Código Mercancía", how="inner")
    if master_base.empty:
        return pd.DataFrame()

    df_ventas = pd.read_parquet(VENTAS_PATH)
    demanda_reciente = engine.compute_daily_demand(df_ventas, recent_days_window=60)

    master = pd.merge(master_base, demanda_reciente, on=["Establecimiento", "Código Mercancía"], how="left")
    master["Demanda_Diaria_Base"] = master["Demanda_Diaria_Base"].fillna(0.0).astype("float32")
    master["Sigma_Empirica"] = master["Sigma_Empirica"].fillna(0.0).astype("float32")

    pareto = engine.compute_pareto(master)
    master = pd.merge(master, pareto, on=["Establecimiento", "Código Mercancía"], how="left")
    master["Clase_ABC"] = master["Clase_ABC"].fillna("C")

    df_calculado = engine.execute_replenishment(master, mes_objetivo=mes_objetivo)

    nombres_tiendas = df_ventas[["Establecimiento", "Nombre Establecimiento"]].drop_duplicates().copy()
    nombres_tiendas["Nombre Establecimiento"] = nombres_tiendas["Nombre Establecimiento"].astype(str).str.strip()

    if "Nombre Establecimiento" in df_calculado.columns:
        df_calculado = df_calculado.drop(columns=["Nombre Establecimiento"])

    df_calculado = pd.merge(df_calculado, nombres_tiendas, on="Establecimiento", how="left")
    col_nombres = df_calculado["Nombre Establecimiento"].astype(str)
    fallback = "SUCURSAL " + df_calculado["Establecimiento"].astype(str)
    df_calculado["Nombre Establecimiento"] = np.where(
        (col_nombres.isna()) | (col_nombres == "nan") | (col_nombres == "None") | (col_nombres == ""),
        fallback,
        col_nombres
    )

    df_calculado["Dias_Cobertura"] = np.where(
        df_calculado["Demanda_Diaria_Base"] > 0,
        np.round(df_calculado["Existencia"] / df_calculado["Demanda_Diaria_Base"], 1),
        99.0
    )

    return df_calculado


# ----------------- SIDEBAR -----------------
if LOGO_PATH and LOGO_PATH.exists():
    logo_base64 = get_base64_encoded_image(LOGO_PATH)
    st.sidebar.markdown(f"""
    <div style="
        background-color: #E51A24; 
        padding: 18px 22px; 
        border-radius: 14px; 
        text-align: center; 
        box-shadow: 0 6px 18px rgba(229, 26, 36, 0.35); 
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        <img src="data:image/png;base64,{logo_base64}" style="max-width: 90%; max-height: 85px; object-fit: contain;">
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background-color: #E51A24; padding: 18px; border-radius: 14px; text-align: center; margin-bottom: 22px;">
        <h2 style="font-weight: 900; color: #FFFFFF; margin: 0; letter-spacing: 1px;">UPPER<span style="color: #111111;">®</span></h2>
        <p style="font-size: 0.75rem; color: #F4F4F4; letter-spacing: 1.5px; font-weight: 700; margin: 0;">BY CANELO ENERGY</p>
    </div>
    """, unsafe_allow_html=True)

mes_actual = datetime.now().month
nombres_meses = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

mes_sel = st.sidebar.selectbox(
    "Mes de Aplicación:",
    options=list(nombres_meses.keys()),
    format_func=lambda x: nombres_meses[x],
    index=mes_actual - 1
)

df_red = load_and_compute_pipeline(mes_objetivo=mes_sel)

if df_red.empty:
    st.error("No se encontraron registros de proveedores Directos coincidentes con existencias.")
    st.stop()

if st.sidebar.button("🔄 Recalcular Memoria", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ----------------- PRINCIPAL -----------------
st.markdown("""
<div class="upper-header">
    <div>
        <h1>UPPER<span>.</span> Pedido sugerido - Proveedores directos</h1>
        <p>Control de Abasto Directo a Tiendas de Conveniencia</p>
    </div>
</div>
""", unsafe_allow_html=True)

tab_tienda, tab_control_tower = st.tabs([
    "🏪 Operación Tienda", 
    "🌐 Reporte Category manager y Coordinadores de zona"
])


# =====================================================================
# PESTAÑA 1: OPERACIÓN TIENDA
# =====================================================================
with tab_tienda:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtro de Tienda (RLS)")
    
    tiendas_disponibles = sorted(df_red["Nombre Establecimiento"].unique())
    tienda_auth = st.sidebar.selectbox("Sucursal Asignada:", options=tiendas_disponibles, index=0)
    
    df_tienda = df_red[df_red["Nombre Establecimiento"] == tienda_auth].copy()
    proveedores_tienda = sorted(df_tienda["Proveedor"].dropna().unique())

    if not proveedores_tienda:
        st.warning("Esta sucursal no registra artículos de proveedores directos.")
    else:
        proveedor_auth = st.sidebar.selectbox("Proveedor Directo:", options=proveedores_tienda)
        df_prov_tienda = df_tienda[df_tienda["Proveedor"] == proveedor_auth].copy()

        st.markdown(f"### Suministro Directo: **{tienda_auth}**")
        st.caption(f"Proveedor: **{proveedor_auth}** | Periodo: **{nombres_meses[mes_sel]}** | Modalidad: **Pieza Suelta**")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        total_skus = len(df_prov_tienda)
        skus_pedir = int((df_prov_tienda["Unidades_Sugeridas"] > 0).sum())
        total_pzas = int(df_prov_tienda["Unidades_Sugeridas"].sum())
        quiebres = int((df_prov_tienda["Estado_Inventario"] == "AGOTADO (Stockout)").sum())

        k1.metric("SKUs en Catálogo", total_skus)
        k2.metric("SKUs con Pedido", skus_pedir)
        k3.metric("Piezas Sugeridas", f"{total_pzas:,}")
        k4.metric("Quiebres de Stock", quiebres, delta=f"-{quiebres}" if quiebres > 0 else "0", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        # Grilla de Captura
        cols_ui = [
            "Código Mercancía", "Descripción", "Nivel 2", "Clase_ABC",
            "Existencia", "Dias_Cobertura", "SS", "MIN", "MAX", "Unidades_Sugeridas", "Estado_Inventario"
        ]
        grid_tienda = df_prov_tienda[cols_ui].copy()
        grid_tienda["Unidades_Autorizadas"] = grid_tienda["Unidades_Sugeridas"]

        grid_edit = st.data_editor(
            grid_tienda,
            column_config={
                "Código Mercancía": st.column_config.NumberColumn("Código", format="%d", disabled=True),
                "Descripción": st.column_config.TextColumn("Descripción", disabled=True),
                "Nivel 2": st.column_config.TextColumn("Categoría", disabled=True),
                "Clase_ABC": st.column_config.TextColumn("Pareto", disabled=True),
                "Existencia": st.column_config.NumberColumn("Stock", format="%.0f", disabled=True),
                "Dias_Cobertura": st.column_config.NumberColumn("Cobertura (Días)", format="%.1f", disabled=True),
                "SS": st.column_config.NumberColumn("SS", format="%.0f", disabled=True),
                "MIN": st.column_config.NumberColumn("Punto Mín", format="%.0f", disabled=True),
                "MAX": st.column_config.NumberColumn("Punto Máx", format="%.0f", disabled=True),
                "Unidades_Sugeridas": st.column_config.NumberColumn("Sugerido", format="%d", disabled=True),
                "Estado_Inventario": st.column_config.TextColumn("Estatus Físico", disabled=True),
                "Unidades_Autorizadas": st.column_config.NumberColumn("Pedido Final", min_value=0, step=1, required=True),
            },
            hide_index=True,
            use_container_width=True
        )

        orden_tienda = grid_edit[grid_edit["Unidades_Autorizadas"] > 0][[
            "Código Mercancía", "Descripción", "Nivel 2", "Unidades_Autorizadas"
        ]].copy()

        if not orden_tienda.empty:
            pzas_finales = int(orden_tienda["Unidades_Autorizadas"].sum())
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label=f"📥 Confirmar y Descargar Orden CSV ({pzas_finales} pzas)",
                data=orden_tienda.to_csv(index=False).encode("utf-8"),
                file_name=f"OC_{tienda_auth.replace(' ', '_')}_{proveedor_auth.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        # ----------------- EXPLICABILIDAD & GRÁFICA LINEAL -----------------
        st.markdown("---")
        st.markdown("#### 🔍 ¿Por qué me sugiere esta cantidad?")
        st.caption("Selecciona cualquier producto para ver su simulación de inventario en el tiempo.")

        skus_disponibles = df_prov_tienda["Descripción"].unique()
        sku_auditar = st.selectbox("Producto a auditar:", options=skus_disponibles)

        fila_sku = df_prov_tienda[df_prov_tienda["Descripción"] == sku_auditar].iloc[0]

        v_demanda = float(fila_sku["Demanda_Diaria_Base"])
        v_estac = float(fila_sku["Factor_Estacional"])
        v_stock = float(fila_sku["Existencia"])
        v_cobertura = float(fila_sku["Dias_Cobertura"])
        v_ss = float(fila_sku["SS"])
        v_min = float(fila_sku["MIN"])
        v_max = float(fila_sku["MAX"])
        v_sug = int(fila_sku["Unidades_Sugeridas"])

        c_diag1, c_diag2, c_diag3 = st.columns(3)
        c_diag1.metric("Venta Diaria Promedio", f"{v_demanda:.1f} pzas/día")
        c_diag2.metric("Cobertura Actual", f"{v_cobertura:.1f} días" if v_demanda > 0 else "Sin Venta")
        c_diag3.metric("Acción Sugerida", f"+{v_sug} piezas" if v_sug > 0 else "0 piezas (No pedir)")

        with st.container():
            st.info(f"**Diagnóstico Operativo:** {sku_auditar}")
            
            if v_sug > 0:
                st.write(
                    f"• **Gatillo de Compra Activado:** Tu inventario físico ({v_stock:.0f} pzas) cayó al nivel de reorden (≤ {v_min:.0f} Mínimo).\n"
                    f"• **Demanda del Ciclo:** Vendes **{v_demanda:.1f} pzas/día**. Durante los 3 días de entrega más los 7 días hasta la siguiente visita, consumirás ~{v_demanda*10:.0f} piezas.\n"
                    f"• **Propósito del Pedido:** Solicitar **{v_sug} piezas** te lleva a la capacidad Máxima ({v_max:.0f} pzas), garantizando abasto total y preservando tu colchón de seguridad de {v_ss:.0f} piezas."
                )
            else:
                st.write(
                    f"• **Stock Suficiente:** Cuentas con {v_stock:.0f} piezas (cobertura para {v_cobertura:.1f} días).\n"
                    f"• Como tu stock está por encima del Mínimo ({v_min:.0f} pzas), **no se requiere compra hoy**. Pedir ahora saturaría la tienda."
                )

        # ----------------- GRÁFICA LINEAL DE INVENTARIO (PLOTLY) -----------------
        st.markdown("##### 📈 Proyección de Inventario a 14 Días (Modelo Dinámico)")

        dias_horizonte = np.arange(0, 15)  # Días 0 al 14
        lead_time = 3

        # 1. Proyección sin pedido (descenso natural)
        stock_sin_pedido = np.maximum(0, v_stock - (dias_horizonte * v_demanda))

        # 2. Proyección con pedido (recibe pedido en Día 3 = LT)
        stock_con_pedido = []
        curr = v_stock
        for d in dias_horizonte:
            if d == lead_time and v_sug > 0:
                curr += v_sug  # Salto de inventario al recibir camión
            curr = max(0, curr - v_demanda)
            stock_con_pedido.append(curr)

        fig = go.Figure()

        # Líneas de Referencia Logísticas
        fig.add_trace(go.Scatter(
            x=[0, 14], y=[v_max, v_max],
            mode="lines", name="Punto Máximo (MAX)",
            line=dict(color="#111111", width=2, dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=[0, 14], y=[v_min, v_min],
            mode="lines", name="Punto Mínimo (Gatillo MIN)",
            line=dict(color="#F59E0B", width=2, dash="dot")
        ))

        fig.add_trace(go.Scatter(
            x=[0, 14], y=[v_ss, v_ss],
            mode="lines", name="Seguridad (SS)",
            line=dict(color="#9CA3AF", width=1.5, dash="dot")
        ))

        # Curva con Pedido (Rojo Upper)
        if v_sug > 0:
            fig.add_trace(go.Scatter(
                x=dias_horizonte, y=stock_con_pedido,
                mode="lines+markers", name=f"Proyección con Pedido (+{v_sug} pzas)",
                line=dict(color="#E51A24", width=3.5),
                marker=dict(size=6)
            ))

        # Curva Sin Pedido (Agotamiento)
        fig.add_trace(go.Scatter(
            x=dias_horizonte, y=stock_sin_pedido,
            mode="lines", name="Trayectoria sin Pedido",
            line=dict(color="#EF4444" if v_sug > 0 else "#2563EB", width=2, dash="dashdot")
        ))

        fig.update_layout(
            title=f"Simulación de Consumo y Reabastecimiento: {sku_auditar}",
            xaxis_title="Días a partir de hoy",
            yaxis_title="Piezas Físicas",
            hovermode="x unified",
            plot_bgcolor="rgba(255,255,255,0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PESTAÑA 2: TORRE DE CONTROL
# =====================================================================
with tab_control_tower:
    st.markdown("### 🌐 Reporte Category manager y Coordinadores de Zona")
    st.caption("Reporte multi-sucursal")

    todos_proveedores = sorted(df_red["Proveedor"].dropna().unique())
    prov_global = st.selectbox("Seleccione Proveedor para Auditoría de Red:", options=todos_proveedores, key="prov_global")

    df_prov_red = df_red[df_red["Proveedor"] == prov_global].copy()

    m1, m2, m3, m4 = st.columns(4)
    tiendas_totales = df_prov_red["Nombre Establecimiento"].nunique()
    total_piezas_red = int(df_prov_red["Unidades_Sugeridas"].sum())
    tiendas_con_quiebre = df_prov_red[df_prov_red["Estado_Inventario"].isin(["AGOTADO (Stockout)", "CRÍTICO (Bajo SS)"])]["Nombre Establecimiento"].nunique()
    pct_salud = round((1 - (tiendas_con_quiebre / tiendas_totales if tiendas_totales > 0 else 0)) * 100, 1)

    m1.metric("Tiendas totales", tiendas_totales)
    m2.metric("Demanda Total Cadena", f"{total_piezas_red:,} pzas")
    m3.metric("Tiendas en Riesgo", tiendas_con_quiebre, delta=f"-{tiendas_con_quiebre}" if tiendas_con_quiebre > 0 else "0", delta_color="inverse")
    m4.metric("Nivel de Servicio Red", f"{pct_salud}%")

    st.markdown("---")
    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        st.markdown("#### Riesgo por Tienda")
        resumen_tiendas = df_prov_red.groupby("Nombre Establecimiento").agg(
            Total_SKUs=("Código Mercancía", "count"),
            SKUs_en_Riesgo=("Estado_Inventario", lambda x: x.isin(["AGOTADO (Stockout)", "CRÍTICO (Bajo SS)"]).sum()),
            Piezas_Requeridas=("Unidades_Sugeridas", "sum"),
            Stock_Total=("Existencia", "sum")
        ).reset_index().sort_values(by=["SKUs_en_Riesgo", "Piezas_Requeridas"], ascending=[False, False])
        
        st.dataframe(
            resumen_tiendas,
            column_config={
                "Nombre Establecimiento": "Sucursal",
                "Total_SKUs": st.column_config.NumberColumn("SKUs"),
                "SKUs_en_Riesgo": st.column_config.NumberColumn("En Peligro"),
                "Piezas_Requeridas": st.column_config.NumberColumn("Pzas Pedir", format="%d"),
                "Stock_Total": st.column_config.NumberColumn("Stock", format="%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )

    with col_der:
        st.markdown("#### Demanda Consolidada por SKU")
        resumen_skus = df_prov_red.groupby(["Código Mercancía", "Descripción"]).agg(
            Tiendas_Afectadas=("Nombre Establecimiento", "nunique"),
            Existencia_Cadena=("Existencia", "sum"),
            Total_Sugerido_Red=("Unidades_Sugeridas", "sum")
        ).reset_index().sort_values(by="Total_Sugerido_Red", ascending=False)

        st.dataframe(
            resumen_skus,
            column_config={
                "Código Mercancía": st.column_config.NumberColumn("Código", format="%d"),
                "Descripción": "Producto",
                "Tiendas_Afectadas": st.column_config.NumberColumn("Tiendas"),
                "Existencia_Cadena": st.column_config.NumberColumn("Stock Red", format="%.0f"),
                "Total_Sugerido_Red": st.column_config.NumberColumn("Total Red (Pzas)", format="%d"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    pedido_maestro = df_prov_red[df_prov_red["Unidades_Sugeridas"] > 0][[
        "Establecimiento", "Nombre Establecimiento", "Código Mercancía", "Descripción", "Existencia", "Unidades_Sugeridas"
    ]].copy()

    if not pedido_maestro.empty:
        csv_maestro = pedido_maestro.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Descargar Pedido Maestro Cadena ({len(pedido_maestro)} entregas)",
            data=csv_maestro,
            file_name=f"PEDIDO_MAESTRO_{prov_global.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Ninguna tienda de la cadena requiere producto de este proveedor actualmente.")
