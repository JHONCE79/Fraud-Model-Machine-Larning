import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Detección de Fraude en Transacciones")

# Agregar un encabezado
st.header("Ingrese los 19 datos solicitados para poder realizar la prediccion")

# Intentar cargar el modelo entrenado
try:
    with open("modelo.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Entradas del usuario
variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
             'k', 'l', 'm',  
             'n', 'o', 'p', 
             'q', 'r', 's', 
             'monto']

# Diccionario para almacenar los datos de entrada
input_data = {}

# Menú desplegable para seleccionar el país
paises = ['AU', 'BR', 'CA', 'CH', 'CL',
          'CO', 'ES', 'FR', 'GB',
          'GT', 'IT', 'KR',
          'MX', 'PT', 'TR',
          'UA', 'US', 'UY']

input_data['pais'] = st.selectbox("Seleccione el país:", paises)

# Crear campos de entrada para cada variable
for var in variables:
    input_data[var] = st.number_input(f"Ingrese valor para {var}:", value=0.0)

if st.button("Predecir"):
    # Crear un DataFrame a partir de los datos recibidos
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Procesar los datos
    input_df.columns = input_df.columns.str.strip().str.lower().str.replace(' ', '')
    
    # Convertir la columna pais a formato j_<pais>
    input_df['j'] = f"j_{input_df['pais'][0]}"
    
    # Eliminar la columna pais, ya que no es necesaria para la predicción
    input_df.drop('pais', axis=1, inplace=True)
    
    # Eliminar la columna 'k', ya que no se considera en el modelo
    input_df.drop('k', axis=1, inplace=True)
    
    # Convertir las columnas a numéricas
    for col in ['q', 'r', 'monto']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Transformación de raíz cuadrada en las variables numéricas (excepto fraude)
    numerical_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df[numerical_cols] = np.sqrt(input_df[numerical_cols])
    
    # Realizar One-Hot Encoding para la variable j
    input_df = pd.get_dummies(input_df, columns=['j'], drop_first=True)
    
    # Asegurar que las columnas estén en el mismo orden que se usó para entrenar el modelo
    model_columns = model.get_booster().feature_names
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Realizar la predicción
    prediction = model.predict(input_df)
    
    # Mostrar el resultado al usuario
    if prediction[0] == 1:
        st.error("La transacción es fraudulenta.")
    else:
        st.success("La transacción no es fraudulenta.")
