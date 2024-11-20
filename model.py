import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import pickle

# Cargar los datos
data = pd.read_csv("data.csv")  # Asegúrate de cargar el dataset que tienes

# Preprocesamiento según lo que hiciste en tu EDA
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# Convertir las columnas 'q', 'r', 'monto' a numéricas (eliminando comas y asegurando que sean float)
# Convertir las columnas 'q', 'r', 'monto' a numéricas (eliminando comas y asegurando que sean float)
data['q'] = pd.to_numeric(data['q'].str.replace(',', ''), errors='coerce')
data['r'] = pd.to_numeric(data['r'].str.replace(',', ''), errors='coerce')
data['monto'] = pd.to_numeric(data['monto'].str.replace(',', ''), errors='coerce')

# Rellenar valores nulos en la columna 'c' usando KNNImputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
data[['c']] = imputer.fit_transform(data[['c']])

# Reemplazo de valores -1 en columnas 'b' y 's' con la mediana
median_b = data.loc[data['b'] != -1, 'b'].median()
median_s = data.loc[data['s'] != -1, 's'].median()
data['b'] = data['b'].replace(-1, median_b)
data['s'] = data['s'].replace(-1, median_s)

# Aplicación de transformación de raíz cuadrada en las variables numéricas
numerical_cols = data.select_dtypes(include=[np.number]).columns.drop('fraude')
data[numerical_cols] = np.sqrt(data[numerical_cols])

# One-hot encoding de la columna 'j'
data = pd.get_dummies(data, columns=['j'], drop_first=True)

# Dividir los datos en variables independientes (X) y dependientes (y)
X = data.drop('fraude', axis=1)
y = data['fraude']

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.24, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Aplicar undersampling para balancear las clases en el entrenamiento
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Crear y entrenar el modelo XGBoost
model = XGBClassifier(
    colsample_bytree=0.6,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=1,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# Guardar el modelo entrenado en un archivo .pkl
with open("modelo.pkl", "wb") as file:
    pickle.dump(model, file)

print("Modelo guardado en 'modelo.pkl'")

expected_features = model.feature_names_in_
print(expected_features)
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar el modelo entrenado
with open("modelo.pkl", "rb") as file:
    model = pickle.load(file)

# Título de la aplicación
st.title("Detección de Fraude en Transacciones")

# Definir las entradas del usuario
variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
             'k', 'l', 'm', 
             'n', 'o', 'p', 
             'q', 'r', 's', 
             'monto']

# Crear un diccionario para almacenar los datos de entrada
input_data = {}

# Menú desplegable para seleccionar el país
paises = ['AU', 'BR', 'CA', 'CH', 'CL',
          'CO', 'ES', 'FR', 'GB',
          'GT', 'IT', 'KR',
          'MX', 'PT', 'TR',
          'UA', 'US', 'UY']

input_data['pais'] = st.selectbox("Seleccione el país:", paises)

# Crear campos de entrada para cada variable (incluyendo pais)
for var in variables:
    input_data[var] = st.number_input(f"Ingrese valor para {var}:", value=0.0)

# Botón para realizar la predicción
if st.button("Predecir"):
    # Crear un DataFrame a partir de los datos recibidos
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Procesar los datos como lo hiciste en tu preprocesamiento
    input_df.columns = input_df.columns.str.strip().str.lower().str.replace(' ', '')
    
    # Convertir la columna pais a formato j_<pais>
    input_df['j'] = f"j_{input_df['pais'][0]}"
    
    # Eliminar la columna pais, ya que no es necesaria para la predicción
    input_df.drop('pais', axis=1, inplace=True)
    
    # Convertir las columnas a numéricas (eliminando comas)
    for col in ['q', 'r', 'monto']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Transformación de raíz cuadrada en las variables numéricas (excepto fraude)
    numerical_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df[numerical_cols] = np.sqrt(input_df[numerical_cols])
    
    # Realizar One-Hot Encoding para la variable j
    input_df = pd.get_dummies(input_df, columns=['j'], drop_first=True)
    
    # Asegurarse que las columnas estén en el mismo orden que se usó para entrenar el modelo
    model_columns = model.get_booster().feature_names
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Realizar la predicción
    prediction = model.predict(input_df)
    
    # Mostrar el resultado al usuario
    if prediction[0] == 1:
        st.error("La transacción es fraudulenta.")
    else:
        st.success("La transacción no es fraudulenta.")
