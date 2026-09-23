import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Reabastecimiento en Tienda",
    page_icon="📦",
    layout="wide",
)


# -----------------------------------------------------------------------------
# 2. GENERACIÓN DE DATOS DUMMIES (MOCKUP SIN COSTOS)
# -----------------------------------------------------------------------------
@st.cache_data
def generar_datos_dummies():
    tiendas = ["UPPER GRANJA", "UPPER CENTRO", "UPPER NORTE"]
    proveedores = ["COCA COLA", "BIMBO", "PEPSICO", "DULCES DE LA ROSA"]

    catalogo_mock = [
        ("COCA COLA", "1001", "COCA COLA NR 600ML", 24),
        ("COCA COLA", "1002", "COCA COLA SIN AZUCAR 600ML", 24),
        ("COCA COLA", "1003", "FANTA NARANJA 600ML", 24),
        ("BIMBO", "2001", "PAN BLANCO GRANDE", 12),
        ("BIMBO", "2002", "DONAS AZUCARADAS 4PZ", 16),
        ("BIMBO", "2003", "NITO 62G", 20),
        ("PEPSICO", "3001", "SABRITAS SAL 160G", 14),
        ("PEPSICO", "3002", "DORITOS NACHO 150G", 14),
        ("PEPSICO", "3003", "EMPERADOR CHOCOLATE 109G", 16),
        ("DULCES DE LA ROSA", "4001", "MAZAPAN 12PZ", 20),
        ("DULCES DE LA ROSA", "4002", "PULPARINDO ORIGINAL 20PZ", 25),
    ]

    filas = []
    np.random.seed(42)

    for tienda in tiendas:
        for prov, cod, desc, pxc in catalogo_mock:
            existencia = int(np.random.choice([0, 2, 5, 12, 25, 40]))
            min_inv = int(np.random.choice([5, 8, 10]))
            max_inv = min_inv + int(np.random.choice([15, 20, 30]))

            necesidad = max(0, max_inv - existencia)

            if existencia <= min_inv:
                cajas_sug = int(np.ceil(necesidad / pxc))
            else:
                cajas_sug = 0

            if existencia == 0 and min_inv > 0:
                estado = "🔴 QUIEBRE DE STOCK"
            elif existencia <= min_inv:
                estado = "🟡 RESURTIDO REQUERIDO"
            elif existencia > max_inv:
                estado = "⚪ SOBREINVENTARIO"
            else:
                estado = "🟢 OK / SUFICIENTE"

            filas.append({
                "Establecimiento": tienda,
                "Proveedor": prov,
                "Código": cod,
                "Descripción": desc,
                "Piezas_Por_Caja": pxc,
                "Existencia": existencia,
                "MIN": min_inv,
                "MAX": max_inv,
                "Cajas_Sugeridas": cajas_sug,
                "Estado_Inventario": estado,
            })

    return pd.DataFrame(filas)


df_datos = generar_datos_dummies()

# -----------------------------------------------------------------------------
# 3. FILTROS LATERALES (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80
)
st.sidebar.title("Control de Recepción")
st.sidebar.markdown("---")

tienda_sel = st.sidebar.selectbox(
    "1. Selecciona la Tienda:",
    options=df_datos["Establecimiento"].unique(),
)

df_tienda = df_datos[df_datos["Establecimiento"] == tienda_sel]

proveedor_sel = st.sidebar.selectbox(
    "2. Selecciona el Proveedor en Mostrador:",
    options=df_tienda["Proveedor"].unique(),
)

df_filtrado = df_tienda[df_tienda["Proveedor"] == proveedor_sel].copy()

# -----------------------------------------------------------------------------
# 4. ENCABEZADO Y KPIS
# -----------------------------------------------------------------------------
st.title("📋 Sugerido de Recepción a Proveedor Directo")
st.subheader(f"Tienda: {tienda_sel} | Proveedor: {proveedor_sel}")

skus_a_resurtir = int((df_filtrado["Cajas_Sugeridas"] > 0).sum())
cajas_iniciales = int(df_filtrado["Cajas_Sugeridas"].sum())
quiebres_iniciales = int(
    (df_filtrado["Estado_Inventario"] == "🔴 QUIEBRE DE STOCK").sum()
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("SKUs del Proveedor", len(df_filtrado))
col2.metric("SKUs a Resurtir", skus_a_resurtir)
col3.metric("SKUs en Quiebre", quiebres_iniciales, delta_color="inverse")
col4.metric("Total Cajas Sugeridas", cajas_iniciales)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. TABLA INTERACTIVA (DATA EDITOR)
# -----------------------------------------------------------------------------
st.markdown("### 🛒 Detalle de Pedido por SKU")
st.caption(
    "Revisa las existencias y ajusta la columna **'Cajas_A_Pedir'** según el físico pactado con el vendedor."
)

df_interfaz = df_filtrado[[
    "Código",
    "Descripción",
    "Piezas_Por_Caja",
    "Existencia",
    "MIN",
    "MAX",
    "Cajas_Sugeridas",
    "Estado_Inventario",
]].copy()

df_interfaz["Cajas_A_Pedir"] = df_interfaz["Cajas_Sugeridas"]

df_editado = st.data_editor(
    df_interfaz,
    column_config={
        "Código": st.column_config.TextColumn("Código", disabled=True),
        "Descripción": st.column_config.TextColumn(
            "Descripción SKU", disabled=True
        ),
        "Piezas_Por_Caja": st.column_config.NumberColumn(
            "Pz x Caja", disabled=True
        ),
        "Existencia": st.column_config.NumberColumn(
            "Exist. Actual", disabled=True
        ),
        "MIN": st.column_config.NumberColumn("MIN", disabled=True),
        "MAX": st.column_config.NumberColumn("MAX", disabled=True),
        "Cajas_Sugeridas": st.column_config.NumberColumn(
            "Sugerido Sistema", disabled=True
        ),
        "Estado_Inventario": st.column_config.TextColumn(
            "Estado", disabled=True
        ),
        "Cajas_A_Pedir": st.column_config.NumberColumn(
            "Cajas Autorizadas",
            help="Modifica las cajas a recibir si hubo ajuste en mostrador",
            min_value=0,
            step=1,
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# 6. RECALCULO Y BOTÓN DE EXPORTACIÓN
# -----------------------------------------------------------------------------
df_editado["Piezas_Totales"] = (
    df_editado["Cajas_A_Pedir"] * df_editado["Piezas_Por_Caja"]
)

total_cajas_final = int(df_editado["Cajas_A_Pedir"].sum())
total_piezas_final = int(df_editado["Piezas_Totales"].sum())

st.markdown("### 📄 Resumen de la Orden Autorizada")
c_res1, c_res2, c_res3 = st.columns([2, 2, 3])
c_res1.metric("Cajas Autorizadas Finales", total_cajas_final)
c_res2.metric("Piezas Totales Autorizadas", total_piezas_final)

csv_orden = df_editado[[
    "Código",
    "Descripción",
    "Piezas_Por_Caja",
    "Existencia",
    "Cajas_A_Pedir",
    "Piezas_Totales",
]].to_csv(index=False)

c_res3.download_button(
    label="⬇️ Descargar Orden de Compra (CSV)",
    data=csv_orden,
    file_name=f"Orden_{tienda_sel.replace(' ', '_')}_{proveedor_sel.replace(' ', '_')}.csv",
    mime="text/csv",
)
