import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from rapidfuzz import process, fuzz

# Configuración inicial
st.set_page_config(page_title="Tracking de Pedidos", layout="wide")

# Título
st.title("Tracking de Pedidos ~ Nissan Parts")

# Sección inicial
st.header("Ingreso de Datos")
via_importacion = st.selectbox("Seleccione la vía de importación:", ["Aérea", "Marítima"])

# Botones principales
consulta_referencia_btn = st.button("Opción: Consulta Referencia", key="consulta_referencia_btn")
busqueda_similar_btn = st.button("Opción: Búsqueda de Nombres Similares", key="busqueda_similar_btn")

# Bandera para mostrar campos adicionales
if "mostrar_referencia" not in st.session_state:
    st.session_state.mostrar_referencia = False

if "mostrar_busqueda_similar" not in st.session_state:
    st.session_state.mostrar_busqueda_similar = False

# Manejo de botones
if consulta_referencia_btn:
    st.session_state.mostrar_referencia = True
    st.session_state.mostrar_busqueda_similar = False

if busqueda_similar_btn:
    st.session_state.mostrar_referencia = False
    st.session_state.mostrar_busqueda_similar = True

# URLs públicas
URL_AEREA = st.secrets["URL_AEREA"]
URL_MARITIMA = st.secrets["URL_MARITIMA"]

gemini_api_key = st.secrets["gemini_api_key"]


# Función para cargar datos
def cargar_datos(url, sheet_name, via):
    df = pd.read_excel(url, sheet_name=sheet_name, header=3)
    df["REFERENCIA"] = df["REFERENCIA"].astype(str)
    df["INVOICE"] = df["INVOICE"].replace(["", "(en blanco)"], pd.NA)
    if via == "Marítima":
        df["SHIP DATE"] = pd.to_datetime(df["SHIP DATE"], errors="coerce")
        df["ETA LA PAZ"] = pd.to_datetime(df["SHIP DATE"] + pd.Timedelta(days=60), errors="coerce")
    df["FECHA LLEGADA"] = pd.to_datetime(df["FECHA LLEGADA"], errors="coerce")
    return df

# Función para validar estado de pedidos
def validar_estado_pedidos(df):
    df["STATUS"] = df["STATUS"].fillna("")
    df["INVOICE"] = df["INVOICE"].replace(["", "(en blanco)"], pd.NA)
    condiciones = [
        df["STATUS"] == "C",
        df["INVOICE"].isnull() & (df["STATUS"] == "B/O"),
        df["FECHA LLEGADA"].notna(),
        df["FECHA LLEGADA"].isna() & df["INVOICE"].notna(),
        (df["FECHA LLEGADA"].isna() & (df["ETA LA PAZ"] < pd.Timestamp.now()) & df["INVOICE"].notna()),
    ]
    resultados = [
        "Cancelado y no será atendido.",
        "Estado en Back Order, posible retraso.",
        "La Pieza ha arribado al almacén.",
        "La Pieza se encuentra en tránsito.",
        "Posible atraso en el pedido.",
    ]
    df["ANALISIS"] = np.select(condiciones, resultados, default="Sin información suficiente.")
    return df

# Función de búsqueda difusa
def buscar_similares(df, columna, termino_busqueda, limite=5, umbral=70):
    termino_busqueda = termino_busqueda.strip().lower()
    df[columna] = df[columna].fillna("").str.lower()
    nombres = df[columna].unique()
    coincidencias = process.extract(termino_busqueda, nombres, scorer=fuzz.partial_ratio, limit=limite)
    nombres_similares = [c[0] for c in coincidencias if c[1] >= umbral]
    return df[df[columna].isin(nombres_similares)]

# Función para análisis con Gemini
def apply_prompt_template(dataframe):
    return f"""
    Eres un asistente de IA especializado en logística. Tu tarea es analizar los resultados de un conjunto de datos ya procesados 
    y proporcionar conclusiones claras y breves.

    ### Instrucciones:
    1. Los datos ya han sido clasificados previamente según su estado en la columna "ANALISIS".
    2. Basa tu análisis en el número de líneas (o pedidos) y en la columna "ANALISIS". No interpretes que hay piezas u otros valores, únicamente considera las líneas de datos presentes.
    3. Tu respuesta debe ser profesional, breve y precisa, evitando incluir información que no se derive explícitamente de los datos del dataframe que tienes disponible.

    ### Objetivo de tu respuesta:
    - Resume el estado de las líneas de datos según los valores únicos en la columna "ANALISIS".
    - Calcula cuántas líneas hay en cada estado (por ejemplo, cuántas están en tránsito, cuántas arribaron al almacén, cuántas fueron canceladas, etc.).
    - No agregues información inventada ni utilices otros valores de las columnas.

    ### Ejemplo de respuesta esperada:
    - "De las 10 líneas de pedido, 8 han arribado al almacén y 2 están en tránsito."
    - "El pedido presenta posibles atrasos en 3 líneas, mientras que 7 ya han arribado."
    - "Todas las líneas de pedido están en tránsito según los datos procesados."

    Con base en la información de la columna "ANALISIS", entrega una conclusión clara y relevante.
    Aplica estas reglas al siguiente conjunto de datos:
    {dataframe[["REFERENCIA", "ANALISIS"]].to_dict()}
    """
def get_gemini_prompt(dataframe):
    prompt = apply_prompt_template(dataframe)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# Mostrar campos según el botón seleccionado
if st.session_state.mostrar_referencia:
    referencia = st.text_input("Ingrese la referencia del pedido:")
    procesar = st.button("Consultar Referencia")
    if procesar and referencia:
        with st.spinner("Procesando referencia..."):
            try:
                # Carga de datos y filtrado por referencia
                if via_importacion == "Aérea":
                    df = cargar_datos(URL_AEREA, "CONTROL_PEDIDOS", "Aérea")
                else:
                    df = cargar_datos(URL_MARITIMA, "CTRL", "Marítima")
                df = validar_estado_pedidos(df)
                df_filtrado = df[df["REFERENCIA"] == referencia]
                if not df_filtrado.empty:
                    st.subheader(f"Resultados para la referencia: {referencia}")
                    st.dataframe(df_filtrado)
                    genai.configure(api_key=gemini_api_key)
                    comentario = get_gemini_prompt(df_filtrado)
                    st.write(comentario)
                else:
                    st.warning("No se encontraron resultados para la referencia proporcionada.")
            except Exception as e:
                st.error(f"Error al procesar la referencia: {e}")

if st.session_state.mostrar_busqueda_similar:
    cliente_busqueda = st.text_input("Ingrese el nombre del cliente para búsqueda:")
    buscar_similares_btn = st.button("Buscar Nombres Similares")
    if buscar_similares_btn and cliente_busqueda.strip():
        with st.spinner("Buscando coincidencias similares..."):
            try:
                # Carga de datos y búsqueda difusa
                if via_importacion == "Aérea":
                    df = cargar_datos(URL_AEREA, "CONTROL_PEDIDOS", "Aérea")
                else:
                    df = cargar_datos(URL_MARITIMA, "CTRL", "Marítima")
                df = validar_estado_pedidos(df)
                resultados_similares = buscar_similares(df, "CLIENTE", cliente_busqueda, limite=10, umbral=80)
                if not resultados_similares.empty:
                    st.subheader(f"Resultados similares para: {cliente_busqueda}")
                    st.dataframe(resultados_similares)                    
                    genai.configure(api_key=gemini_api_key)
                    comentario = get_gemini_prompt(resultados_similares)
                    st.write(comentario)
                else:
                    st.warning("No se encontraron coincidencias similares.")
            except Exception as e:
                st.error(f"Error durante la búsqueda de nombres similares: {e}")
