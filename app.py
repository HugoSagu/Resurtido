from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from engine import ReplenishmentEngine

st.set_page_config(page_title="Portal DSD | Pedido Sugerido", layout="wide")

engine = ReplenishmentEngine(lead_time_days=3, review_frequency_days=7, extra_coverage_days=2, min_cv=0.50)

@st.cache_data(ttl=None)
def load_and_compute_pipeline(mes_objetivo: int) -> pd.DataFrame:
    df_cat = pd.read_parquet("catalogo.parquet")
    df_cat["Tipo de proveedor"] = df_cat["Tipo de proveedor"].astype(str).str.strip().str.upper()
    df_cat_dsd = df_cat[df_cat["Tipo de proveedor"] == "DIRECTO"].copy()

    df_exist = pd.read_parquet("existencias.parquet")
    cols_exist = [c for c in ["Establecimiento", "Código Mercancía", "Existencia", "Descripción"] if c in df_exist.columns]
    df_exist = df_exist[cols_exist].copy()

    df_estac = pd.read_parquet("estacionalidad.parquet")
    df_estac_mes = df_estac[df_estac["Mes"] == mes_objetivo][["Nivel 2", "Factor_Estacional"]].copy()

    df_cat_estac = pd.merge(df_cat_dsd, df_estac_mes, on="Nivel 2", how="left")
    df_cat_estac["Factor_Estacional"] = df_cat_estac["Factor_Estacional"].fillna(1.0).astype("float32")

    master_base = pd.merge(df_cat_estac, df_exist, on="Código Mercancía", how="inner")
    if master_base.empty:
        return pd.DataFrame()

    df_ventas = pd.read_parquet("ventas.parquet")
    demanda_reciente = engine.compute_daily_demand(df_ventas, recent_days_window=60)

    master = pd.merge(master_base, demanda_reciente, on=["Establecimiento", "Código Mercancía"], how="left")
    master["Demanda_Diaria_Base"] = master["Demanda_Diaria_Base"].fillna(0.0).astype("float32")
    master["Sigma_Empirica"] = master["Sigma_Empirica"].fillna(0.0).astype("float32")

    pareto = engine.compute_pareto(master)
    master = pd.merge(master, pareto, on=["Establecimiento", "Código Mercancía"], how="left")
    master["Clase_ABC"] = master["Clase_ABC"].fillna("C")

    df_calculado = engine.execute_replenishment(master, mes_objetivo=mes_objetivo)

# 7. Asignar Nombres de Establecimiento desde Ventas de forma segura
    nombres_tiendas = df_ventas[["Establecimiento", "Nombre Establecimiento"]].drop_duplicates().copy()
    
    # Des-categorizar explícitamente a texto para evitar el bloqueo de Pandas
    nombres_tiendas["Nombre Establecimiento"] = nombres_tiendas["Nombre Establecimiento"].astype(str).str.strip()

    # Si la columna ya existía en df_calculado, la eliminamos para evitar duplicados _x, _y
    if "Nombre Establecimiento" in df_calculado.columns:
        df_calculado = df_calculado.drop(columns=["Nombre Establecimiento"])

    # Fusionar nombres de sucursales
    df_calculado = pd.merge(df_calculado, nombres_tiendas, on="Establecimiento", how="left")

    # Imputación segura con NumPy (vectorizada y sin restricciones categóricas)
    col_nombre = df_calculado["Nombre Establecimiento"].astype(str)
    fallback = "SUCURSAL " + df_calculado["Establecimiento"].astype(str)
    
    df_calculado["Nombre Establecimiento"] = np.where(
        (col_nombre.isna()) | (col_nombre == "nan") | (col_nombre == "None") | (col_nombre == ""),
        fallback,
        col_nombre
    )

    return df_calculado

st.sidebar.title("Control de Acceso DSD")
mes_actual = datetime.now().month
nombres_meses = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
mes_seleccionado = st.sidebar.selectbox("Mes de Aplicación:", options=list(nombres_meses.keys()), format_func=lambda x: nombres_meses[x], index=mes_actual - 1)

df_master = load_and_compute_pipeline(mes_objetivo=mes_seleccionado)

if df_master.empty:
    st.error("No se encontraron registros de proveedores Directos coincidentes con existencias.")
    st.stop()

tiendas = sorted(df_master["Nombre Establecimiento"].unique())
tienda_auth = st.sidebar.selectbox("Sucursal (Identidad):", options=tiendas, index=0)
df_tienda = df_master[df_master["Nombre Establecimiento"] == tienda_auth].copy()

proveedores = sorted(df_tienda["Proveedor"].dropna().unique())
proveedor_auth = st.sidebar.selectbox("Proveedor Directo:", options=proveedores)
df_proveedor = df_tienda[df_tienda["Proveedor"] == proveedor_auth].copy()

st.title(f"Reabastecimiento DSD: {tienda_auth}")
st.caption(f"Proveedor: **{proveedor_auth}** | Suministro en Pieza Suelta")

k1, k2, k3, k4 = st.columns(4)
k1.metric("SKUs en Catálogo", len(df_proveedor))
k2.metric("SKUs a Pedir", int((df_proveedor["Unidades_Sugeridas"] > 0).sum()))
k3.metric("Piezas Sugeridas", f"{int(df_proveedor['Unidades_Sugeridas'].sum()):,}")
k4.metric("Quiebres de Stock", int((df_proveedor["Estado_Inventario"] == 'AGOTADO (Stockout)').sum()))

st.divider()

cols_ui = ["Código Mercancía", "Descripción", "Nivel 2", "Clase_ABC", "Existencia", "SS", "MIN", "MAX", "Unidades_Sugeridas", "Estado_Inventario"]
grid_base = df_proveedor[cols_ui].copy()
grid_base["Unidades_Autorizadas"] = grid_base["Unidades_Sugeridas"]

st.subheader("Captura y Autorización")
grid_editado = st.data_editor(
    grid_base,
    column_config={
        "Código Mercancía": st.column_config.NumberColumn("Código", format="%d", disabled=True),
        "Descripción": st.column_config.TextColumn("Descripción", disabled=True),
        "Existencia": st.column_config.NumberColumn("Stock", format="%.0f", disabled=True),
        "Unidades_Autorizadas": st.column_config.NumberColumn("Pedido Final", min_value=0, step=1, required=True),
    },
    hide_index=True,
    use_container_width=True
)

orden = grid_editado[grid_editado["Unidades_Autorizadas"] > 0][["Código Mercancía", "Descripción", "Nivel 2", "Unidades_Autorizadas"]].copy()

if not orden.empty:
    st.download_button(
        label=f"📥 Descargar Orden ({int(orden['Unidades_Autorizadas'].sum())} pzas)",
        data=orden.to_csv(index=False).encode("utf-8"),
        file_name=f"OC_{tienda_auth}_{proveedor_auth}.csv",
        mime="text/csv"
    )
