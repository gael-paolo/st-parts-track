import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import gcsfs  # Librería para interactuar con GCP

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
    Eres un asistente de IA especializado en logística. Tu tarea es analizar los resultados de un conjunto de datos ya procesados y proporcionar conclusiones claras y breves.

    ### Instrucciones:
    1. Los datos ya han sido clasificados previamente según su estado.
    2. Solo debes interpretar los valores en la columna "ANALISIS", que describe el estado de cada pieza.
    3. Tu respuesta debe ser breve y profesional, resumiendo la situación de los pedidos en **1 o 2 oraciones** como máximo.

    ### Objetivo de tu respuesta:
    - Proporciona una conclusión clara sobre el estado de los pedidos, destacando arribos, piezas en tránsito, cancelaciones o atrasos.
    - Evita listar los datos de la tabla directamente, ya que el usuario ya los tiene disponibles en pantalla.
    - Si es relevante, menciona cuántos pedidos están en cada estado.

    ### Ejemplo de respuesta esperada:
    - "De las 10 piezas solicitadas, 8 han arribado al almacén, 1 está en tránsito y 1 fue cancelada."
    - "El pedido presenta posibles atrasos, ya que 3 piezas no han llegado en la fecha estimada."
    - "Todas las piezas han arribado al almacén según las fechas registradas."

    Aplica estas reglas al siguiente conjunto de datos:
    {dataframe[["REFERENCIA", "ANALISIS"]].to_dict()}
    """

def get_gemini_prompt(dataframe):
    prompt = apply_prompt_template(dataframe)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def validar_estado_pedidos(df):
    # Normalizar las columnas clave para evitar errores
    df["FECHA LLEGADA"] = pd.to_datetime(df["FECHA LLEGADA"], errors="coerce")
    df["ETA LA PAZ"] = pd.to_datetime(df["ETA LA PAZ"], errors="coerce")
    df["STATUS"] = df["STATUS"].fillna("")
    df["INVOICE"] = df["INVOICE"].fillna("")

    # Condiciones evaluadas en orden de prioridad
    cancelado = df["STATUS"] == "C"
    ha_arribado = df["FECHA LLEGADA"].notnull()
    en_transito = df["FECHA LLEGADA"].isnull() & df["INVOICE"].notnull()
    atrasado = (df["FECHA LLEGADA"].isnull()) & (df["ETA LA PAZ"] < pd.Timestamp.now())

    # Condiciones y resultados para la columna ANALISIS
    condiciones = [cancelado, ha_arribado, en_transito, atrasado]
    resultados = [
        "Cancelado y no será atendido.",
        "Ha arribado al almacén.",
        "Está en tránsito.",
        "Posible atraso en el pedido.",
    ]

    # Crear o actualizar la columna "ANALISIS"
    df["ANALISIS"] = np.select(condiciones, resultados, default="Sin información suficiente.")

# Sección 2: Tabla con los resultados del pedido
if procesar and referencia:
    try:
        if via_importacion == "Aérea":
            air = pd.read_excel(URL_AEREA, sheet_name="CONTROL_PEDIDOS", header=3)
            air["REFERENCIA"] = air["REFERENCIA"].astype(str)
            air["INVOICE"] = air["INVOICE"].astype(str)
            air = air.iloc[:, :-1]
            air['FECHA LLEGADA'] = pd.to_datetime(air['FECHA LLEGADA'], errors='coerce')
            air['ETA LA PAZ'] = pd.to_datetime(air['ETA LA PAZ'], errors='coerce').fillna('-')

            # Validar pedidos y actualizar la columna "ANALISIS" en el mismo DataFrame
            validar_estado_pedidos(air)

            df_filtrado = air[air["REFERENCIA"] == referencia]
        else:
            sea = pd.read_excel(URL_MARITIMA, sheet_name="CTRL", header=3)
            sea["REFERENCIA"] = sea["REFERENCIA"].astype(str)
            sea["INVOICE"] = sea["INVOICE"].astype(str)
            sea = sea.iloc[:, :-1]
            sea['SHIP DATE'] = pd.to_datetime(sea['SHIP DATE'], errors='coerce')
            sea['FECHA LLEGADA'] = pd.to_datetime(sea['FECHA LLEGADA'], errors='coerce')
            sea['ETA LA PAZ'] = sea['SHIP DATE'] + pd.Timedelta(days=60)
            sea['ETA LA PAZ'] = sea['ETA LA PAZ'].fillna('-')

            # Validar pedidos y actualizar la columna "ANALISIS" en el mismo DataFrame
            validar_estado_pedidos(sea)

            df_filtrado = sea[sea["REFERENCIA"] == referencia]

        if not df_filtrado.empty:
            st.subheader(f"Resultados para la referencia: {referencia}")
            st.dataframe(df_filtrado)
        else:
            st.warning("No se encontraron resultados para la referencia ingresada.")
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")

# Sección 3: Análisis con la API Gemini
if procesar and df_filtrado is not None and not df_filtrado.empty:
    try:
        genai.configure(api_key=gemini_api_key)
        comentario = get_gemini_prompt(df_filtrado)
        st.header("Análisis")
        st.write(comentario)
    except Exception as e:
        st.error(f"Error al conectar con la API Gemini: {e}")
