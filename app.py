"""
Sistema de Reabastecimiento Dinámico DSD con Explicabilidad y Torre de Control.
Branding: UPPER by Canelo Energy | Frontend: Streamlit.
"""
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from engine import ReplenishmentEngine

# ----------------- CONFIGURACIÓN DE PÁGINA -----------------
st.set_page_config(
    page_title="UPPER | Reabastecimiento DSD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- ESTILOS PERSONALIZADOS (UPPER BRANDING) -----------------
st.markdown("""
<style>
    /* Importar fuente moderna y limpia */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    /* Colores Principales Upper */
    :root {
        --upper-red: #E51A24;
        --upper-red-hover: #C4121B;
        --upper-dark: #121212;
        --upper-gray-bg: #F8F9FA;
        --upper-card-border: #E5E7EB;
    }

    /* Barra Superior / Header decorativo */
    .upper-header {
        background: linear-gradient(90deg, #111111 0%, #1F1F1F 70%, #E51A24 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    .upper-header h1 {
        font-weight: 900;
        letter-spacing: 1.5px;
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2.2rem;
    }
    .upper-header span {
        color: #E51A24;
    }
    .upper-header p {
        color: #CCCCCC;
        margin-top: 6px;
        font-size: 0.95rem;
        margin-bottom: 0;
    }

    /* Tarjetas de Métricas (KPI Cards) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid var(--upper-card-border);
        border-left: 5px solid var(--upper-red);
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(229, 26, 36, 0.12);
    }
    div[data-testid="stMetricLabel"] p {
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #555555 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #111111 !important;
        font-weight: 800 !important;
    }

    /* Botón de Descarga / Acción Primaria */
    .stDownloadButton > button {
        background-color: var(--upper-red) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(229, 26, 36, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: var(--upper-red-hover) !important;
        box-shadow: 0 6px 18px rgba(229, 26, 36, 0.45) !important;
        transform: translateY(-1px);
    }

    /* Pestañas (Tabs) Estilo Upper */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 700;
        font-size: 1rem;
        padding: 10px 20px;
        color: #555555;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: var(--upper-red) !important;
        border-bottom: 3px solid var(--upper-red) !important;
        background-color: transparent !important;
    }

    /* Sidebar Estilizada */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E5E7EB;
    }

    /* Cajas de Diagnóstico / Alertas */
    .stAlert {
        border-radius: 8px;
        border-left: 5px solid var(--upper-red) !important;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- INICIALIZAR MOTOR Y RUTAS -----------------
engine = ReplenishmentEngine(
    lead_time_days=3,
    review_frequency_days=7,
    extra_coverage_days=2,
    min_cv=0.50
)

BASE_DIR = Path(__file__).resolve().parent

VENTAS_PATH = BASE_DIR / "ventas.parquet"
CATALOGO_PATH = BASE_DIR / "catalogo.parquet"
EXISTENCIAS_PATH = BASE_DIR / "existencias.parquet"
ESTACIONALIDAD_PATH = BASE_DIR / "estacionalidad.parquet"

ARCHIVOS_REQUERIDOS = [VENTAS_PATH, CATALOGO_PATH, EXISTENCIAS_PATH, ESTACIONALIDAD_PATH]

if not all(p.is_file() for p in ARCHIVOS_REQUERIDOS):
    st.error("🚨 Faltan bases de datos Parquet en el directorio del proyecto.")
    st.stop()


# ----------------- PIPELINE DE CÁLCULO VECTORIZADO -----------------
@st.cache_data(ttl=None, show_spinner="Calculando reabastecimiento Upper...")
def load_and_compute_pipeline(mes_objetivo: int) -> pd.DataFrame:
    # 1. Catálogo Directo
    df_cat = pd.read_parquet(CATALOGO_PATH)
    df_cat["Tipo de proveedor"] = df_cat["Tipo de proveedor"].astype(str).str.strip().str.upper()
    df_cat_dsd = df_cat[df_cat["Tipo de proveedor"] == "DIRECTO"].copy()

    if df_cat_dsd.empty:
        return pd.DataFrame()

    # 2. Existencias (Purga estricta de costos y precios)
    df_exist = pd.read_parquet(EXISTENCIAS_PATH)
    cols_exist = [c for c in ["Establecimiento", "Código Mercancía", "Existencia", "Descripción"] if c in df_exist.columns]
    df_exist = df_exist[cols_exist].copy()

    # 3. Factor Estacional
    df_estac = pd.read_parquet(ESTACIONALIDAD_PATH)
    df_estac_mes = df_estac[df_estac["Mes"] == mes_objetivo][["Nivel 2", "Factor_Estacional"]].copy()

    df_cat_estac = pd.merge(df_cat_dsd, df_estac_mes, on="Nivel 2", how="left")
    df_cat_estac["Factor_Estacional"] = df_cat_estac["Factor_Estacional"].fillna(1.0).astype("float32")

    # Cruce con Existencias
    master_base = pd.merge(df_cat_estac, df_exist, on="Código Mercancía", how="inner")
    if master_base.empty:
        return pd.DataFrame()

    # 4. Demanda reciente (60 días)
    df_ventas = pd.read_parquet(VENTAS_PATH)
    demanda_reciente = engine.compute_daily_demand(df_ventas, recent_days_window=60)

    master = pd.merge(
        master_base,
        demanda_reciente,
        on=["Establecimiento", "Código Mercancía"],
        how="left"
    )
    master["Demanda_Diaria_Base"] = master["Demanda_Diaria_Base"].fillna(0.0).astype("float32")
    master["Sigma_Empirica"] = master["Sigma_Empirica"].fillna(0.0).astype("float32")

    # 5. Pareto ABC
    pareto = engine.compute_pareto(master)
    master = pd.merge(master, pareto, on=["Establecimiento", "Código Mercancía"], how="left")
    master["Clase_ABC"] = master["Clase_ABC"].fillna("C")

    # 6. Motor Min-Max
    df_calculado = engine.execute_replenishment(master, mes_objetivo=mes_objetivo)

    # 7. Asignar Nombres de Tiendas
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

    # 8. Días de Cobertura
    df_calculado["Dias_Cobertura"] = np.where(
        df_calculado["Demanda_Diaria_Base"] > 0,
        np.round(df_calculado["Existencia"] / df_calculado["Demanda_Diaria_Base"], 1),
        99.0
    )

    return df_calculado


# ----------------- PARÁMETROS GLOBALES (SIDEBAR) -----------------
st.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 15px;">
    <h2 style="font-weight: 900; color: #111111; margin-bottom: 0;">UPPER<span style="color: #E51A24;">®</span></h2>
    <p style="font-size: 0.75rem; color: #777777; letter-spacing: 1px; font-weight: 600;">BY CANELO ENERGY</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros Generales")

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


# ----------------- HEADER PRINCIPAL UPPER -----------------
st.markdown("""
<div class="upper-header">
    <h1>UPPER<span>.</span> REABASTECIMIENTO DSD</h1>
    <p>Motor de Control Logístico y Abasto Directo a Tiendas de Conveniencia</p>
</div>
""", unsafe_allow_html=True)

# ----------------- PESTAÑAS PRINCIPALES -----------------
tab_tienda, tab_control_tower = st.tabs([
    "🏪 Operación Tienda", 
    "🌐 Torre de Control de Red"
])


# =====================================================================
# PESTAÑA 1: VISTA DE TIENDA
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

        # KPIs con estilo Card
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
        st.markdown("#### Captura y Autorización de Pedido")
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

        # ----------------- EXPLICABILIDAD -----------------
        st.markdown("---")
        st.markdown("#### 🔍 Auditoría: ¿Por qué me sugiere esta cantidad?")
        st.caption("Selecciona cualquier producto para ver el desglose en lenguaje natural.")

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
        v_clase = str(fila_sku["Clase_ABC"])

        c_diag1, c_diag2, c_diag3 = st.columns(3)
        c_diag1.metric("Venta Diaria Promedio", f"{v_demanda:.1f} pzas/día")
        c_diag2.metric("Cobertura Actual", f"{v_cobertura:.1f} días" if v_demanda > 0 else "Sin Venta")
        c_diag3.metric("Acción Sugerida", f"+{v_sug} piezas" if v_sug > 0 else "0 piezas (No pedir)")

        with st.container():
            st.info(f"**Radiografía Operativa:** {sku_auditar}")
            
            st.write(f"1. **Ritmo de Venta:** Vendes en promedio **{v_demanda:.1f} piezas al día** (últimos 60 días).")
            
            pct_estac = round((v_estac - 1.0) * 100, 1)
            if pct_estac > 0:
                st.write(f"2. **Temporada Upper:** 🔥 **Alta demanda.** Incremento del **+{pct_estac}%** en {nombres_meses[mes_sel]}.")
            elif pct_estac < 0:
                st.write(f"2. **Temporada Upper:** ❄️ **Baja demanda.** Ajuste estacional de **{pct_estac}%** en {nombres_meses[mes_sel]}.")
            else:
                st.write(f"2. **Temporada Upper:** ⚖️ **Demanda neutra.** Comportamiento regular.")

            if v_stock <= 0:
                st.write("3. **Situación en Tienda:** 🚨 **Agotado (Stockout).** No hay inventario en exhibición.")
            elif v_cobertura <= 3:
                st.write(f"3. **Situación en Tienda:** ⚠️ **Riesgo crítico.** Con **{v_stock:.0f} piezas** te quedan solo **{v_cobertura:.1f} días** de venta. El camión tarda 3 días en llegar.")
            elif v_cobertura <= 10:
                st.write(f"3. **Situación en Tienda:** 🟡 **Cobertura estándar.** Abasto para **{v_cobertura:.1f} días**, entrarías a usar tu reserva.")
            else:
                st.write(f"3. **Situación en Tienda:** 🟢 **Stock holgado.** Inventario suficiente para **{v_cobertura:.1f} días**.")

            if v_sug > 0:
                st.success(
                    f"📌 **Conclusión:** Se solicitan **{v_sug} piezas** porque el stock actual ({v_stock:.0f}) está por debajo del Punto Mínimo ({v_min:.0f}). "
                    f"Con este pedido se alcanza el Punto Máximo ({v_max:.0f}), cubriendo 10 días de ciclo y protegiendo tu reserva de seguridad de {v_ss:.0f} piezas."
                )
            else:
                st.success(
                    f"📌 **Conclusión:** No se solicita producto hoy porque el stock actual ({v_stock:.0f} pzas) supera el Punto Mínimo ({v_min:.0f} pzas). "
                    f"Pedir más producto generaría sobrealmacenamiento innecesario."
                )

        st.markdown("##### 📊 Termómetro de Niveles Físicos")
        df_termo = pd.DataFrame({
            "Nivel": [
                "1. Stock Físico",
                "2. Seguridad (SS)",
                "3. Punto Mínimo",
                "4. Capacidad Máxima",
                "5. Post-Pedido"
            ],
            "Piezas": [v_stock, v_ss, v_min, v_max, v_stock + v_sug]
        })
        st.bar_chart(data=df_termo.set_index("Nivel"), horizontal=True, color="#E51A24")


# =====================================================================
# PESTAÑA 2: TORRE DE CONTROL (SUPERVISIÓN)
# =====================================================================
with tab_control_tower:
    st.markdown("### 🌐 Torre de Control de Red")
    st.caption("Visión macro multi-sucursal para Category Managers y Supervisión")

    todos_proveedores = sorted(df_red["Proveedor"].dropna().unique())
    prov_global = st.selectbox("Seleccione Proveedor para Auditoría de Red:", options=todos_proveedores, key="prov_global")

    df_prov_red = df_red[df_red["Proveedor"] == prov_global].copy()

    m1, m2, m3, m4 = st.columns(4)
    tiendas_totales = df_prov_red["Nombre Establecimiento"].nunique()
    total_piezas_red = int(df_prov_red["Unidades_Sugeridas"].sum())
    tiendas_con_quiebre = df_prov_red[df_prov_red["Estado_Inventario"].isin(["AGOTADO (Stockout)", "CRÍTICO (Bajo SS)"])]["Nombre Establecimiento"].nunique()
    pct_salud = round((1 - (tiendas_con_quiebre / tiendas_totales if tiendas_totales > 0 else 0)) * 100, 1)

    m1.metric("Tiendas de la Red", tiendas_totales)
    m2.metric("Demanda Total Cadena", f"{total_piezas_red:,} pzas")
    m3.metric("Tiendas en Riesgo", tiendas_con_quiebre, delta=f"-{tiendas_con_quiebre}" if tiendas_con_quiebre > 0 else "0", delta_color="inverse")
    m4.metric("Nivel de Servicio Red", f"{pct_salud}%")

    st.markdown("---")
    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        st.markdown("#### 🚨 Semáforo de Riesgo por Tienda")
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
        st.markdown("#### 📦 Necesidad Consolidada por SKU")
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
