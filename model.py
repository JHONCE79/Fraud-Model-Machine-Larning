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
