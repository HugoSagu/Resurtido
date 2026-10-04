"""
Sistema de Reabastecimiento Dinámico DSD para Tiendas de Conveniencia.
Frontend: Streamlit | Motor: Min-Max con Estacionalidad por Nivel 2 y Pareto Local.
"""
import streamlit as st
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from engine import ReplenishmentEngine

# ----------------- CONFIGURACIÓN DE INTERFAZ -----------------
st.set_page_config(
    page_title="Portal DSD | Pedido Sugerido",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar motor de cálculo (LT=3, R=7, Cobertura=2, CV_min=0.50)
engine = ReplenishmentEngine(
    lead_time_days=3,
    review_frequency_days=7,
    extra_coverage_days=2,
    min_cv=0.50
)

# ----------------- VALIDACIÓN DE ARCHIVOS -----------------
ARCHIVOS_REQUERIDOS = [
    "ventas.parquet",
    "catalogo.parquet",
    "existencias.parquet",
    "estacionalidad.parquet"
]

def validar_archivos_base() -> bool:
    return all(Path(f).is_file() for f in ARCHIVOS_REQUERIDOS)

if not validar_archivos_base():
    st.error("🚨 Faltan bases de datos en formato Parquet en la raíz del proyecto.")
    st.info("Asegúrate de haber generado: `ventas.parquet`, `catalogo.parquet`, `existencias.parquet` y `estacionalidad.parquet`.")
    st.stop()


# ----------------- CARGA DE DATOS OPTIMIZADA (CACHEADA) -----------------
@st.cache_data(ttl=None, show_spinner="Procesando modelos de demanda y Min-Max...")
def load_and_compute_pipeline(mes_objetivo: int) -> pd.DataFrame:
    """
    Carga columnas operativas desde Parquet, cruza catálogos DSD, existencias,
    aplica estacionalidad del mes y ejecuta el motor Min-Max.
    """
    # 1. Cargar Catálogo (Solo Proveedores Directos)
    df_cat = pd.read_parquet("catalogo.parquet")
    df_cat["Tipo de proveedor"] = df_cat["Tipo de proveedor"].astype(str).str.strip().str.upper()
    df_cat_dsd = df_cat[df_cat["Tipo de proveedor"] == "DIRECTO"].copy()

    if df_cat_dsd.empty:
        return pd.DataFrame()

    # 2. Cargar Existencias y purgar posibles columnas monetarias remanentes
    df_exist = pd.read_parquet("existencias.parquet")
    cols_exist_utiles = ["Establecimiento", "Código Mercancía", "Existencia", "Descripción"]
    df_exist = df_exist[[c for c in cols_exist_utiles if c in df_exist.columns]].copy()

    # 3. Cargar Estacionalidad precalculada para el mes destino
    df_estac = pd.read_parquet("estacionalidad.parquet")
    df_estac_mes = df_estac[df_estac["Mes"] == mes_objetivo][["Nivel 2", "Factor_Estacional"]].copy()

    # Cruce Catálogo + Estacionalidad
    df_cat_estac = pd.merge(df_cat_dsd, df_estac_mes, on="Nivel 2", how="left")
    df_cat_estac["Factor_Estacional"] = df_cat_estac["Factor_Estacional"].fillna(1.0).astype("float32")

    # Cruce con Existencias por SKU
    master_base = pd.merge(
        df_cat_estac,
        df_exist,
        on="Código Mercancía",
        how="inner"
    )

    if master_base.empty:
        return pd.DataFrame()

    # 4. Cargar Ventas para Demanda Reciente (Ventana de 60 días)
    df_ventas = pd.read_parquet("ventas.parquet")
    demanda_reciente = engine.compute_daily_demand(df_ventas, recent_days_window=60)

    # Cruce con la Demanda Base Diaria
    master = pd.merge(
        master_base,
        demanda_reciente,
        on=["Establecimiento", "Código Mercancía"],
        how="left"
    )
    master["Demanda_Diaria_Base"] = master["Demanda_Diaria_Base"].fillna(0.0).astype("float32")
    master["Sigma_Empirica"] = master["Sigma_Empirica"].fillna(0.0).astype("float32")

    # 5. Pareto ABC a nivel Tienda + Nivel 2
    pareto = engine.compute_pareto(master)
    master = pd.merge(master, pareto, on=["Establecimiento", "Código Mercancía"], how="left")
    master["Clase_ABC"] = master["Clase_ABC"].fillna("C")

    # 6. Ejecución Min-Max
    df_calculado = engine.execute_replenishment(master, mes_objetivo=mes_objetivo)

    # 7. Asignar Nombres de Establecimiento desde Ventas si falta
    nombres_tiendas = df_ventas[["Establecimiento", "Nombre Establecimiento"]].drop_duplicates()
    df_calculado = pd.merge(df_calculado, nombres_tiendas, on="Establecimiento", how="left")
    df_calculado["Nombre Establecimiento"] = df_calculado["Nombre Establecimiento"].fillna(
        "SUCURSAL " + df_calculado["Establecimiento"].astype(str)
    )

    return df_calculado


# ----------------- BARRA LATERAL: CONTROL Y SEGURIDAD (RLS) -----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081986.png", width=70)
st.sidebar.title("Control de Acceso")

# Selector de Mes de Proyección (por defecto el mes actual)
mes_actual = datetime.now().month
nombres_meses = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

mes_seleccionado = st.sidebar.selectbox(
    "Mes de Aplicación (Estacionalidad):",
    options=list(nombres_meses.keys()),
    format_func=lambda x: nombres_meses[x],
    index=mes_actual - 1
)

# Cargar datos cacheados
df_master = load_and_compute_pipeline(mes_objetivo=mes_seleccionado)

if df_master.empty:
    st.error("No se encontraron registros de proveedores Directos que crucen con existencias.")
    st.stop()

# Simulación de Identidad de Tienda (Row Level Security estricto)
tiendas_disponibles = sorted(df_master["Nombre Establecimiento"].unique())
tienda_sesion = st.sidebar.selectbox(
    "Sucursal Activa (Identidad):",
    options=tiendas_disponibles,
    index=0
)

# FILTRO RLS EN MEMORIA: La vista jamás consulta datos de otra sucursal
df_tienda = df_master[df_master["Nombre Establecimiento"] == tienda_sesion].copy()

# Selector de Proveedor Directo
proveedores = sorted(df_tienda["Proveedor"].dropna().unique())
if not proveedores:
    st.warning("Esta sucursal no tiene artículos asignados a proveedores directos.")
    st.stop()

proveedor_activo = st.sidebar.selectbox("Proveedor Directo (DSD):", options=proveedores)

# Filtrar por proveedor seleccionado
df_proveedor = df_tienda[df_tienda["Proveedor"] == proveedor_activo].copy()

st.sidebar.divider()
if st.sidebar.button("🔄 Refrescar Memoria de Cálculo"):
    st.cache_data.clear()
    st.rerun()


# ----------------- DASHBOARD OPERATIVO DE TIENDA -----------------
st.title(f"Reabastecimiento DSD: {tienda_sesion}")
st.caption(
    f"Proveedor: **{proveedor_activo}** | Mes: **{nombres_meses[mes_seleccionado]}** | "
    f"Política: **LT=3 días, R=7 días (Piezas Sueltas)**"
)

# Indicadores Físicos
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_skus = len(df_proveedor)
skus_reorden = int((df_proveedor["Unidades_Sugeridas"] > 0).sum())
total_piezas = int(df_proveedor["Unidades_Sugeridas"].sum())
quiebres_stock = int((df_proveedor["Estado_Inventario"] == "AGOTADO (Stockout)").sum())

kpi1.metric("SKUs en Catálogo", total_skus)
kpi2.metric("SKUs que Requieren Pedido", skus_reorden)
kpi3.metric("Piezas Totales Sugeridas", f"{total_piezas:,}")
kpi4.metric("Quiebres de Stock", quiebres_stock, delta=f"-{quiebres_stock}" if quiebres_stock > 0 else "0", delta_color="inverse")

st.divider()

# ----------------- GRILLA DE AUTORIZACIÓN (SIN COSTOS NI PRECIOS) -----------------
st.subheader("Captura y Autorización de Pedido")
st.markdown("Valida físicamente el inventario en anaquel/bodega y ajusta el pedido final si existen limitaciones de exhibición.")

columnas_ui = [
    "Código Mercancía",
    "Descripción",
    "Nivel 2",
    "Clase_ABC",
    "Existencia",
    "SS",
    "MIN",
    "MAX",
    "Unidades_Sugeridas",
    "Estado_Inventario"
]

tabla_edicion = df_proveedor[columnas_ui].copy()
tabla_edicion["Unidades_Autorizadas"] = tabla_edicion["Unidades_Sugeridas"]

# Editor de datos con columnas protegidas y solo 'Unidades_Autorizadas' editable
grid_resultado = st.data_editor(
    tabla_edicion,
    column_config={
        "Código Mercancía": st.column_config.NumberColumn("Código", format="%d", disabled=True),
        "Descripción": st.column_config.TextColumn("Descripción", disabled=True),
        "Nivel 2": st.column_config.TextColumn("Categoría", disabled=True),
        "Clase_ABC": st.column_config.TextColumn("Pareto", disabled=True),
        "Existencia": st.column_config.NumberColumn("Stock Actual", format="%.0f", disabled=True),
        "SS": st.column_config.NumberColumn("Seguridad (SS)", format="%.0f", disabled=True),
        "MIN": st.column_config.NumberColumn("Punto Mín", format="%.0f", disabled=True),
        "MAX": st.column_config.NumberColumn("Punto Máx", format="%.0f", disabled=True),
        "Unidades_Sugeridas": st.column_config.NumberColumn("Sugerido (Pzas)", format="%d", disabled=True),
        "Estado_Inventario": st.column_config.TextColumn("Estatus Físico", disabled=True),
        "Unidades_Autorizadas": st.column_config.NumberColumn(
            "Pedido Final (Pzas)",
            help="Modifica las piezas a solicitar según espacio disponible",
            min_value=0,
            step=1,
            required=True
        ),
    },
    hide_index=True,
    use_container_width=True
)

# ----------------- GENERACIÓN DE ORDEN DE COMPRA DSD -----------------
orden_final = grid_resultado[grid_resultado["Unidades_Autorizadas"] > 0][[
    "Código Mercancía", "Descripción", "Nivel 2", "Unidades_Autorizadas"
]].copy()

st.markdown("### Resumen de la Orden")
if orden_final.empty:
    st.info("No hay piezas programadas para entrega en este proveedor.")
else:
    piezas_finales = int(orden_final["Unidades_Autorizadas"].sum())
    skus_finales = len(orden_final)
    st.success(f"Listo para emitir orden: **{skus_finales} productos** sumando **{piezas_finales:,} piezas**.")

    archivo_csv = orden_final.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Descargar Orden DSD a Proveedor (.CSV)",
        data=archivo_csv,
        file_name=f"ORDEN_{tienda_sesion.replace(' ', '_')}_{proveedor_activo.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
