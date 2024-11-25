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
gemini_api_key = st.text_input("Ingresa tu API key de Gemini", type="password")

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
    Eres un asistente de IA especializado en logística. ...
    {dataframe[["REFERENCIA", "ANALISIS"]].to_dict()}
    """

def get_gemini_prompt(dataframe):
    prompt = apply_prompt_template(dataframe)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def validar_estado_pedidos(df):
    ha_arribado = df["FECHA LLEGADA"].notnull()
    en_transito = df["FECHA LLEGADA"].isnull() & df["INVOICE"].notnull()
    cancelado = df["STATUS"] == "C"
    atrasado = (df["FECHA LLEGADA"].isnull()) & (pd.to_datetime(df["ETA LA PAZ"], errors="coerce") < pd.Timestamp.now())
    condiciones = [ha_arribado, en_transito, cancelado, atrasado]
    resultados = [
        "Ha arribado al almacén.",
        "Está en tránsito.",
        "Cancelado y no será atendido.",
        "Posible atraso en el pedido.",
    ]
    df["ANALISIS"] = np.select(condiciones, resultados, default="Sin información suficiente.")
    return df

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
            df_filtrado = air[air["REFERENCIA"] == referencia]
        else:
            sea = pd.read_excel(URL_MARITIMA, sheet_name="CTRL", header=3)
            sea["REFERENCIA"] = sea["REFERENCIA"].astype(str)
            sea["INVOICE"] = sea["INVOICE"].astype(str)
            sea = sea.iloc[:, :-1]
            sea['SHIP DATE'] = pd.to_datetime(sea['SHIP DATE'], errors='coerce')
            sea['ETA LA PAZ'] = sea['SHIP DATE'] + pd.Timedelta(days=60)
            sea['ETA LA PAZ'] = sea['ETA LA PAZ'].fillna('-')
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
    df_validado = validar_estado_pedidos(df_filtrado)
    try:
        genai.configure(api_key=gemini_api_key)
        comentario = get_gemini_prompt(df_validado)
        st.header("Análisis")
        st.write(comentario)
    except Exception as e:
        st.error(f"Error al conectar con la API Gemini: {e}")
