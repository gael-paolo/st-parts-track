import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import gcsfs

# Configuración inicial de la aplicación
st.set_page_config(page_title="Tracking de Pedidos", layout="wide")

# Título de la aplicación
st.title("Tracking de Pedidos ~ Nissan Parts")

# Sección 1: Ingreso de datos
st.header("Ingreso de Datos")

# Campo para seleccionar la vía de importación
via_importacion = st.selectbox("Seleccione la vía de importación:", ["Aérea", "Marítima"])

# Campo para ingresar la referencia
referencia = st.text_input("Ingrese la referencia del pedido:")

# Campo para ingresar la API key de Gemini
gemini_api_key = st.secrets["gemini_api_key"]

# Botón para procesar los datos
procesar = st.button("Procesar")

# Variables para almacenar los DataFrames según la vía de importación
df_filtrado = None

# URLs públicas de los archivos en GCP
URL_AEREA = "https://storage.googleapis.com/bk_parts/Maestro%20Pedidos%20Especificos.xlsx"
URL_MARITIMA = "https://storage.googleapis.com/bk_parts/Maestro%20Pedidos%20Stock.xlsx"

# Función para generar el prompt para Gemini

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

def validar_estado_pedidos(df):
    df["FECHA LLEGADA"] = pd.to_datetime(df["FECHA LLEGADA"], errors="coerce")
    df["ETA LA PAZ"] = pd.to_datetime(df["ETA LA PAZ"], errors="coerce")
    df["STATUS"] = df["STATUS"].fillna("")
    df["INVOICE"] = df["INVOICE"].replace(["", "(en blanco)"], pd.NA)

    cancelado = df["STATUS"] == "C"
    back_order = df["INVOICE"].isnull() & (df["STATUS"] == "B/O")
    ha_arribado = df["FECHA LLEGADA"].notna()
    en_transito = df["FECHA LLEGADA"].isna() & df["INVOICE"].notna()
    atrasado = (
        df["FECHA LLEGADA"].isna() & 
        (df["ETA LA PAZ"] < pd.Timestamp.now()) & 
        df["INVOICE"].notna())

    condiciones = [cancelado, back_order, ha_arribado, en_transito, atrasado]
    resultados = [
        "Cancelado y no será atendido.",
        "Estado en Back Order, posible retraso.",
        "La Pieza ha arribado al almacén.",
        "La Pieza se encuentra en tránsito.",
        "Posible atraso en el pedido."]

    df["ANALISIS"] = np.select(condiciones, resultados, default="Sin información suficiente.")

# Sección 2: Tabla con los resultados del pedido
if procesar and referencia:
    try:
        if via_importacion == "Aérea":
            # Cargar el archivo correspondiente
            air = pd.read_excel(URL_AEREA, sheet_name="CONTROL_PEDIDOS", header=3)
            air["REFERENCIA"] = air["REFERENCIA"].astype(str)
            air["INVOICE"] = air["INVOICE"].replace(["", "(en blanco)"], pd.NA)
            air = air.iloc[:, :-1]
            air["FECHA LLEGADA"] = pd.to_datetime(air["FECHA LLEGADA"], errors="coerce")
            air["ETA LA PAZ"] = pd.to_datetime(air["ETA LA PAZ"], errors="coerce").fillna('-')

            # Validar pedidos
            validar_estado_pedidos(air)

            # Filtrar por referencia
            df_filtrado = air[air["REFERENCIA"] == referencia]
            df_filtrado["FECHA LLEGADA"] = df_filtrado["FECHA LLEGADA"].dt.strftime("%Y-%m-%d")
            df_filtrado["ETA LA PAZ"] = df_filtrado["ETA LA PAZ"].dt.strftime("%Y-%m-%d")

        else:
            # Cargar el archivo correspondiente
            sea = pd.read_excel(URL_MARITIMA, sheet_name="CTRL", header=3)
            sea["REFERENCIA"] = sea["REFERENCIA"].astype(str)
            sea["INVOICE"] = sea["INVOICE"].replace(["", "(en blanco)"], pd.NA)
            sea = sea.iloc[:, :-1]
            sea['SHIP DATE'] = pd.to_datetime(sea['SHIP DATE'], errors='coerce')
            sea['FECHA LLEGADA'] = pd.to_datetime(sea['FECHA LLEGADA'], errors='coerce')
            sea['ETA LA PAZ'] = sea['SHIP DATE'] + pd.Timedelta(days=60)
            sea['ETA LA PAZ'] = pd.to_datetime(sea['ETA LA PAZ'], errors='coerce').fillna('-')

            # Validar pedidos
            validar_estado_pedidos(sea)

            # Filtrar por referencia
            df_filtrado = sea[sea["REFERENCIA"] == referencia]
            df_filtrado["FECHA LLEGADA"] = df_filtrado["FECHA LLEGADA"].dt.strftime("%Y-%m-%d")
            df_filtrado["ETA LA PAZ"] = df_filtrado["ETA LA PAZ"].dt.strftime("%Y-%m-%d")
            df_filtrado["SHIP DATE"] = df_filtrado["SHIP DATE"].dt.strftime("%Y-%m-%d")

        # Mostrar resultados
        if not df_filtrado.empty:
            st.subheader(f"Resultados para la referencia: {referencia}")
            st.dataframe(df_filtrado.drop(df_filtrado.columns[5], axis=1))

    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")

# Sección 3: Análisis con la API Gemini
if procesar and df_filtrado is not None and not df_filtrado.empty:
    try:
        genai.configure(api_key=gemini_api_key)
        comentario = get_gemini_prompt(df_filtrado)
        st.write(comentario)
    except Exception as e:
        st.error(f"Error al conectar con la API Gemini: {e}")
