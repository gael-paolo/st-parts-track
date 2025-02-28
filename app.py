import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from rapidfuzz import process, fuzz
import unidecode

# Configuración inicial
st.set_page_config(page_title="Tracking de Pedidos", layout="wide")

# Título
st.title("Tracking de Pedidos ~ Nissan Parts")

# Sección Reservas
st.header("Consulta Pedidos Reservados")
via_importacion = st.selectbox("Seleccione la vía de importación:", ["Aérea", "Marítima"])

# Botones principales
consulta_referencia_btn = st.button("Opción: Consulta Referencia", key="consulta_referencia_btn")
busqueda_similar_btn = st.button("Opción: Búsqueda de Nombres Similares", key="busqueda_similar_btn")

# Bandera para mostrar campos adicionales
if "mostrar_referencia" not in st.session_state:
    st.session_state.mostrar_referencia = False

if "mostrar_busqueda_similar" not in st.session_state:
    st.session_state.mostrar_busqueda_similar = False

if "mostrar_transito" not in st.session_state:
    st.session_state.mostrar_transito = False

# Manejo de botones
if consulta_referencia_btn:
    st.session_state.mostrar_referencia = True
    st.session_state.mostrar_busqueda_similar = False

if busqueda_similar_btn:
    st.session_state.mostrar_referencia = False
    st.session_state.mostrar_busqueda_similar = True

# URLs públicas
URL_AEREA = "https://storage.googleapis.com/bk_parts/air_report_tracking.csv"
URL_MARITIMA = "https://storage.googleapis.com/bk_parts/sea_report_tracking.csv"
URL_TRANSITO = "https://storage.googleapis.com/bk_parts/transit_report.csv"

#URL_AEREA = st.secrets["URL_AEREA"]
#URL_MARITIMA = st.secrets["URL_MARITIMA"]
#URL_TRANSITO = st.secrets["URL_TRANSITO"]

gemini_api_key = "AIzaSyC8w2d70IXIZi9jqZLwEtZUIUBnom1ttZs"
#gemini_api_key = st.secrets["gemini_api_key"]


# Función para cargar datos
def cargar_datos(url, via):
    df = pd.read_csv(url)
    df["REFERENCIA"] = df["REFERENCIA"].astype(str)
    df["INVOICE"] = df["INVOICE"].replace(["", "(en blanco)"], pd.NA)
    if via == "Marítima":
        df["SHIP_DATE"] = pd.to_datetime(df["SHIP_DATE"], errors="coerce")
        df["ETA_LP"] = pd.to_datetime(df["SHIP_DATE"] + pd.Timedelta(days=60), errors="coerce")
        df["SHIP_DATE"] = df["SHIP_DATE"].apply(
            lambda x: pd.NaT if pd.isnull(x) or x == pd.Timestamp("1900-01-01") else x)
    df["FECHA_LLEGADA"] = pd.to_datetime(df["FECHA_LLEGADA"], errors="coerce")
    df["ETA_LP"] = pd.to_datetime(df["ETA_LP"], errors="coerce")
    return df

def cargar_transito(url):
    df = pd.read_csv(url)
    return df

# Función para validar estado de pedidos
def validar_estado_pedidos(df):
    # Limpieza de las columnas STATUS e INVOICE
    df["STATUS"] = df["STATUS"].fillna("")
    df["INVOICE"] = df["INVOICE"].replace(["", "(en blanco)"], pd.NA)
    df["INVOICE"] = df["INVOICE"].apply(
        lambda x: pd.NA if pd.isnull(x) or x == "Sin Invoice" else x)

    # Considerar valores "1900-01-01" como NaT en las fechas
    df["FECHA_LLEGADA"] = df["FECHA_LLEGADA"].apply(
        lambda x: pd.NaT if pd.isnull(x) or x == pd.Timestamp("1900-01-01") else x)
    df["ETA_LP"] = df["ETA_LP"].apply(
        lambda x: pd.NaT if pd.isnull(x) or x == pd.Timestamp("1900-01-01") or x == pd.Timestamp("1900-03-02") else x)

    # Definir las condiciones para el análisis
    condiciones = [
        (df["STATUS"] == "C") | (df["STATUS"] == "U"),
        (df["FECHA_LLEGADA"].isna() & (df["ETA_LP"] < pd.Timestamp.now()) & df["INVOICE"].isnull()),
        (df["FECHA_LLEGADA"].isna() & (df["ETA_LP"] < pd.Timestamp.now()) & df["INVOICE"].notna()),
        df["INVOICE"].isnull() & (df["STATUS"] == "B/O"),
        df["FECHA_LLEGADA"].notna(),
        df["FECHA_LLEGADA"].isna() & df["INVOICE"].notna()
        ]

    # Resultados correspondientes a las condiciones
    resultados = [
        "Cancelado y no será atendido.",
        "Pedido sin Atención y Retrasado",
        "Pedido Retrasado en tránsito",
        "Estado en Back Order, posible retraso.",
        "La Pieza ha arribado al almacén.",
        "La Pieza se encuentra en tránsito."
    ]
    # Aplicar las condiciones y asignar resultados al campo ANALISIS
    df["ANALISIS"] = np.select(condiciones, resultados, default="Sin información suficiente.")
    return df

# Función de búsqueda difusa
def limpiar_texto(texto):
    """Normaliza el texto: minúsculas, sin acentos, sin caracteres especiales."""
    if pd.isna(texto):
        return ""
    return unidecode.unidecode(texto.lower().strip())

def buscar_similares(df, columna, termino_busqueda, limite=5, umbral=80):
    """
    Busca nombres similares en la columna dada del DataFrame usando coincidencia difusa mejorada.
    """
    termino_busqueda = limpiar_texto(termino_busqueda)
    df[columna] = df[columna].fillna("").apply(limpiar_texto)
    nombres = df[columna].unique()
    coincidencias = process.extract(termino_busqueda, nombres, scorer=fuzz.token_sort_ratio, limit=limite)
    mejores_coincidencias = [c[0] for c in coincidencias if c[1] >= umbral]    
    return df[df[columna].isin(mejores_coincidencias)]

# Función para análisis con Gemini
def apply_prompt_template(dataframe):
    return f"""
    Eres un asistente de IA especializado en logística. Tu tarea es analizar los resultados de un conjunto de datos ya procesados 
    y proporcionar conclusiones claras y detalladas.

    ### Instrucciones:
    1. Los datos ya han sido clasificados previamente según su estado en la columna "ANALISIS".
    2. Basa tu análisis en el número de líneas (o pedidos), en la columna "ANALISIS" y en las Columnas de Fechas entregadas. 
    3. Proporciona no solo un resumen cuantitativo, sino también un análisis cualitativo que evalúe posibles patrones importantes observados en los datos.
    4. Importante: Si consideras que no hay  más información adicional para analizar, se puntual con la respuesta. NO lo hagas muy amplio
    5. Importante: Si solamente existe una línea, no hace falta que hagas más análisis mas que entregar el estado.
    5. Tu respuesta debe ser profesional, detallada y relevante, evitando información no fundamentada en los datos proporcionados.

    ### Objetivo de tu respuesta:
    - Resume el estado de las líneas de datos según los valores únicos en la columna "ANALISIS".
    - Calcula cuántas líneas hay en cada estado (por ejemplo, cuántas están en tránsito, cuántas arribaron al almacén, cuántas fueron canceladas, etc.).
    - Proporciona un análisis adicional: ¿existen algún dato que vale la pena mencionar? 
   
    ### Ejemplo de respuesta esperada:
    - "De las 10 líneas de pedido, 8 han arribado al almacén y 2 están en tránsito. Sin embargo, se observa que los tiempos de tránsito promedio superan los 60 días en el 20% de los casos, lo que podría sugerir la necesidad de revisar las rutas logísticas. Considerar procesos de seguimiento más estrictos podría ser beneficioso."
    - "El pedido presenta posibles atrasos en 3 líneas, mientras que 7 ya han arribado. Este retraso parece estar asociado a una falta de documentación en el proveedor. Una recomendación sería revisar los procesos de gestión de documentos para reducir estos tiempos en el futuro."
    - "Se detecta que el 30% de las líneas de pedido se encuentran en tránsito prolongado (más de 90 días). Esto podría tener impacto en el nivel de servicio al cliente. Recomendamos implementar un sistema de monitoreo más preciso para identificar los puntos críticos en el transporte."

    Con base en la información de la columna "ANALISIS", entrega una conclusión clara, relevante y con observaciones adicionales.

    ### Datos para analizar:
    {dataframe[["REFERENCIA", "ANALISIS"]].to_dict()}
    """
def get_gemini_prompt(dataframe):
    prompt = apply_prompt_template(dataframe)
    model = genai.GenerativeModel("gemini-2.0-flash")
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
                    df = cargar_datos(URL_AEREA, "Aérea")
                else:
                    df = cargar_datos(URL_MARITIMA, "Marítima")
                df_filtrado = df[df["REFERENCIA"] == referencia]
                df_filtrado = validar_estado_pedidos(df_filtrado)
                df_filtrado["ETA_LP"] = pd.to_datetime(df_filtrado["ETA_LP"]).dt.strftime("%Y/%m/%d")
                df_filtrado["FECHA_LLEGADA"] = pd.to_datetime(df_filtrado["FECHA_LLEGADA"]).dt.strftime("%Y/%m/%d")

                if not df_filtrado.empty:
                    st.subheader(f"Resultados para la referencia: {referencia}")
                    st.dataframe(df_filtrado.drop(df_filtrado.columns[5], axis=1))
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
                    df = cargar_datos(URL_AEREA, "Aérea")
                else:
                    df = cargar_datos(URL_MARITIMA, "Marítima")
                df = validar_estado_pedidos(df)
                df["ETA_LP"] = pd.to_datetime(df["ETA_LP"]).dt.strftime("%Y/%m/%d")
                df["FECHA_LLEGADA"] = pd.to_datetime(df["FECHA_LLEGADA"]).dt.strftime("%Y/%m/%d")
                resultados_similares = buscar_similares(df, "CLIENTE", cliente_busqueda, limite=10, umbral=80)

                if not resultados_similares.empty:
                    st.subheader(f"Resultados similares para: {cliente_busqueda}")
                    st.dataframe(resultados_similares.drop(resultados_similares.columns[5], axis=1))                    
                    genai.configure(api_key=gemini_api_key)
                    comentario = get_gemini_prompt(resultados_similares)
                    st.write(comentario)
                else:
                    st.warning("No se encontraron coincidencias similares.")
            except Exception as e:
                st.error(f"Error durante la búsqueda de nombres similares: {e}")

st.header("Consulta Material en Tránsito")
transito_btn = st.button("Opción: Consulta Material en Tránsito", key="consulta_transito_btn")

if transito_btn:
    st.session_state.mostrar_referencia = False
    st.session_state.mostrar_busqueda_similar = False
    st.session_state.mostrar_transito = True

if st.session_state.mostrar_transito:
    NP = st.text_input("Ingrese el NP que requieres:")
    procesar_t = st.button("Consultar Tránsito")
    if procesar_t and NP:
        with st.spinner("Procesando NP..."):
            try:
                df = cargar_transito(URL_TRANSITO)
                df_filtrado = df[df["NP"] == NP]
                df_filtrado["INVOICE"] = df_filtrado["INVOICE"].astype(str)
                # Carga de datos y filtrado por referencia
                if not df_filtrado.empty:
                    st.subheader("Resultados:")
                    st.dataframe(df_filtrado)
                else:
                    st.warning("No se encontraron resultados para el NP Proporcionado.")
            except Exception as e:
                st.error(f"Error al procesar: {e}")
