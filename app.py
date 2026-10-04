"""
Sistema de Reabastecimiento Dinámico DSD con Tablero de Control para Supervisores.
Frontend: Streamlit | Arquitectura: Doble Perfil (Tienda / Supervisión de Red).
"""
from datetime import datetime
from pathlib import Path
import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
from engine import ReplenishmentEngine

# ----------------- CONFIGURACIÓN DE PÁGINA -----------------
st.set_page_config(
    page_title="Portal proveedores directos",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar motor de cálculo
engine = ReplenishmentEngine(
    lead_time_days=3,
    review_frequency_days=7,
    extra_coverage_days=2,
    min_cv=0.50
)

# ----------------- VALIDACIÓN DE BASES PARQUET -----------------
ARCHIVOS_REQUERIDOS = [
    "ventas.parquet",
    "catalogo.parquet",
    "existencias.parquet",
    "estacionalidad.parquet"
]

if not all(Path(f).is_file() for f in ARCHIVOS_REQUERIDOS):
    st.error("🚨 Faltan bases de datos Parquet en la raíz del proyecto.")
    st.stop()


# ----------------- PIPELINE DE CÁLCULO GLOBAL -----------------
@st.cache_data(ttl=None, show_spinner="Procesando red completa de tiendas y modelos Min-Max...")
def load_and_compute_pipeline(mes_objetivo: int) -> pd.DataFrame:
    # 1. Catálogo Directo
    df_cat = pd.read_parquet("catalogo.parquet")
    df_cat["Tipo de proveedor"] = df_cat["Tipo de proveedor"].astype(str).str.strip().str.upper()
    df_cat_dsd = df_cat[df_cat["Tipo de proveedor"] == "DIRECTO"].copy()

    if df_cat_dsd.empty:
        return pd.DataFrame()

    # 2. Existencias
    df_exist = pd.read_parquet("existencias.parquet")
    cols_exist = [c for c in ["Establecimiento", "Código Mercancía", "Existencia", "Descripción"] if c in df_exist.columns]
    df_exist = df_exist[cols_exist].copy()

    # 3. Factor Estacional por Nivel 2
    df_estac = pd.read_parquet("estacionalidad.parquet")
    df_estac_mes = df_estac[df_estac["Mes"] == mes_objetivo][["Nivel 2", "Factor_Estacional"]].copy()

    df_cat_estac = pd.merge(df_cat_dsd, df_estac_mes, on="Nivel 2", how="left")
    df_cat_estac["Factor_Estacional"] = df_cat_estac["Factor_Estacional"].fillna(1.0).astype("float32")

    # Cruce con existencias
    master_base = pd.merge(df_cat_estac, df_exist, on="Código Mercancía", how="inner")
    if master_base.empty:
        return pd.DataFrame()

    # 4. Demanda reciente (60 días)
    df_ventas = pd.read_parquet("ventas.parquet")
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
    fallback = "SUCURSAL " + df_calculado["Establecimiento"].astype(str)
    col_nombres = df_calculado["Nombre Establecimiento"].astype(str)
    df_calculado["Nombre Establecimiento"] = np.where(
        (col_nombres.isna()) | (col_nombres == "nan") | (col_nombres == "None") | (col_nombres == ""),
        fallback,
        col_nombres
    )

    # Días de cobertura
    df_calculado["Dias_Cobertura"] = np.where(
        df_calculado["Demanda_Diaria_Base"] > 0,
        np.round(df_calculado["Existencia"] / df_calculado["Demanda_Diaria_Base"], 1),
        99.0
    )

    return df_calculado


# ----------------- PARÁMETROS GLOBALES (SIDEBAR) -----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081986.png", width=65)
st.sidebar.title("Configuración Global")

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

st.sidebar.divider()
if st.sidebar.button("🔄 Recalcular Memoria"):
    st.cache_data.clear()
    st.rerun()


# ----------------- ESTRUCTURA PRINCIPAL POR PESTAÑAS -----------------
tab_tienda, tab_control_tower = st.tabs([
    "🏪 Operación Tienda (Encargado)", 
    "🌐 Operaciones (Category manager y Coordinadores)"
])


# =====================================================================
# PESTAÑA 1: VISTA DE TIENDA (ENFOQUE OPERATIVO LOCAL)
# =====================================================================
with tab_tienda:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtro de Tienda")
    
    tiendas_disponibles = sorted(df_red["Nombre Establecimiento"].unique())
    tienda_auth = st.sidebar.selectbox("Sucursal Asignada:", options=tiendas_disponibles, index=0)
    
    # RLS en memoria
    df_tienda = df_red[df_red["Nombre Establecimiento"] == tienda_auth].copy()
    
    proveedores_tienda = sorted(df_tienda["Proveedor"].dropna().unique())
    if not proveedores_tienda:
        st.warning("Esta sucursal no registra artículos de proveedores directos.")
    else:
        proveedor_auth = st.sidebar.selectbox("Proveedor Directo:", options=proveedores_tienda)
        df_prov_tienda = df_tienda[df_tienda["Proveedor"] == proveedor_auth].copy()

        st.subheader(f"Suministro Directo: {tienda_auth}")
        st.caption(f"Proveedor: **{proveedor_auth}** | Mes: **{nombres_meses[mes_sel]}**")

        k1, k2, k3, k4 = st.columns(4)
        total_skus = len(df_prov_tienda)
        skus_pedir = int((df_prov_tienda["Unidades_Sugeridas"] > 0).sum())
        total_pzas = int(df_prov_tienda["Unidades_Sugeridas"].sum())
        quiebres = int((df_prov_tienda["Estado_Inventario"] == "AGOTADO (Stockout)").sum())

        k1.metric("SKUs en Catálogo", total_skus)
        k2.metric("SKUs a Pedir", skus_pedir)
        k3.metric("Piezas Sugeridas", f"{total_pzas:,}")
        k4.metric("Quiebres de Stock", quiebres, delta=f"-{quiebres}" if quiebres > 0 else "0", delta_color="inverse")

        # Tabla editable
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
                "Clase_ABC": st.column_config.TextColumn("ABC", disabled=True),
                "Existencia": st.column_config.NumberColumn("Stock", format="%.0f", disabled=True),
                "Dias_Cobertura": st.column_config.NumberColumn("Cobertura (Días)", format="%.1f", disabled=True),
                "SS": st.column_config.NumberColumn("SS", format="%.0f", disabled=True),
                "MIN": st.column_config.NumberColumn("MIN", format="%.0f", disabled=True),
                "MAX": st.column_config.NumberColumn("MAX", format="%.0f", disabled=True),
                "Unidades_Sugeridas": st.column_config.NumberColumn("Sugerido", format="%d", disabled=True),
                "Estado_Inventario": st.column_config.TextColumn("Estatus", disabled=True),
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
            c_desc, c_wsp = st.columns(2)
            
            with c_desc:
                st.download_button(
                    label=f"📥 Descargar CSV ({pzas_finales} pzas)",
                    data=orden_tienda.to_csv(index=False).encode("utf-8"),
                    file_name=f"OC_{tienda_auth.replace(' ', '_')}_{proveedor_auth.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            with c_wsp:
                msg = f"*PEDIDO DSD - {tienda_auth}*\n*Proveedor:* {proveedor_auth}\n\n"
                for _, r in orden_tienda.iterrows():
                    msg += f"• {r['Descripción']}: *{r['Unidades_Autorizadas']} pzas*\n"
                link_wsp = f"https://api.whatsapp.com/send?text={urllib.parse.quote(msg)}"
                st.link_button("📲 Enviar por WhatsApp", link_wsp)


# =====================================================================
# PESTAÑA 2: SUPERVISIÓN DE RED COMPLETA
# =====================================================================
with tab_control_tower:
    st.title("🌐 operaciones (Category manager y Coordinadores regionales)")
    st.caption("Visión consolidada multi-tienda para Category Managers y Dirección de Operaciones")

    # Selector de Proveedor Global para análisis
    todos_proveedores = sorted(df_red["Proveedor"].dropna().unique())
    prov_global = st.selectbox("Seleccione Proveedor para Auditoría de Red:", options=todos_proveedores, key="prov_global")

    df_prov_red = df_red[df_red["Proveedor"] == prov_global].copy()

    # Métricas Globales de la Cadena
    m1, m2, m3, m4 = st.columns(4)
    tiendas_totales = df_prov_red["Nombre Establecimiento"].nunique()
    total_piezas_red = int(df_prov_red["Unidades_Sugeridas"].sum())
    tiendas_con_quiebre = df_prov_red[df_prov_red["Estado_Inventario"].isin(["AGOTADO (Stockout)", "CRÍTICO (Bajo SS)"])]["Nombre Establecimiento"].nunique()
    pct_salud = round((1 - (tiendas_con_quiebre / tiendas_totales if tiendas_totales > 0 else 0)) * 100, 1)

    m1.metric("Tiendas Auditadas", tiendas_totales)
    m2.metric("Demanda Total de la Red", f"{total_piezas_red:,} pzas")
    m3.metric("Tiendas con inventario en déficit", tiendas_con_quiebre, delta=f"-{tiendas_con_quiebre}" if tiendas_con_quiebre > 0 else "0", delta_color="inverse")
    m4.metric("Nivel de Servicio de Red", f"{pct_salud}%")

    st.markdown("---")

    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        st.subheader("🚨 Semáforo de Riesgo por Tienda")
        st.caption("Tiendas con mayor concentración de quiebres o stock crítico para este proveedor")
        
        resumen_tiendas = df_prov_red.groupby("Nombre Establecimiento").agg(
            Total_SKUs=("Código Mercancía", "count"),
            SKUs_en_Riesgo=("Estado_Inventario", lambda x: x.isin(["AGOTADO (Stockout)", "CRÍTICO (Bajo SS)"]).sum()),
            Piezas_Requeridas=("Unidades_Sugeridas", "sum"),
            Stock_Total=("Existencia", "sum")
        ).reset_index()

        resumen_tiendas = resumen_tiendas.sort_values(by=["SKUs_en_Riesgo", "Piezas_Requeridas"], ascending=[False, False])
        
        st.dataframe(
            resumen_tiendas,
            column_config={
                "Nombre Establecimiento": "Sucursal",
                "Total_SKUs": st.column_config.NumberColumn("SKUs Catálogo"),
                "SKUs_en_Riesgo": st.column_config.NumberColumn("SKUs en Peligro", help="En Quiebre o Bajo Inventario de Seguridad"),
                "Piezas_Requeridas": st.column_config.NumberColumn("Total Pzas a Pedir", format="%d"),
                "Stock_Total": st.column_config.NumberColumn("Existencia Física", format="%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )

    with col_der:
        st.subheader("📦 Consolidado de Necesidad por SKU")
        st.caption("Volumen total que debe surtir el proveedor para cubrir toda la cadena")
        
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
                "Tiendas_Afectadas": st.column_config.NumberColumn("Tiendas Activas"),
                "Existencia_Cadena": st.column_config.NumberColumn("Stock en Red", format="%.0f"),
                "Total_Sugerido_Red": st.column_config.NumberColumn("Demanda Red (Pzas)", format="%d"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📑 Descarga del Pedido Maestro Consolidado")
    st.markdown("Genera el archivo matriz con el desglose tienda por tienda para enviar a la dirección de ventas del proveedor.")

    pedido_maestro = df_prov_red[df_prov_red["Unidades_Sugeridas"] > 0][[
        "Establecimiento", "Nombre Establecimiento", "Código Mercancía", "Descripción", "Existencia", "Unidades_Sugeridas"
    ]].copy()

    if not pedido_maestro.empty:
        csv_maestro = pedido_maestro.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Descargar Pedido Maestro Nacional ({len(pedido_maestro)} órdenes de entrega)",
            data=csv_maestro,
            file_name=f"PEDIDO_MAESTRO_{prov_global.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Ninguna tienda de la cadena requiere producto de este proveedor en este momento.")
