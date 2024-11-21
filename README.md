# Detección de Fraude en Transacciones

Este proyecto utiliza un modelo de Machine Learning basado en XGBoost para predecir si una transacción es fraudulenta o no, a partir de un conjunto de características de la transacción.


## Realizado por:

- John Steven Ceballos Agudelo

## Descripción

Este sistema permite predecir si una transacción es fraudulenta usando un modelo de XGBoost previamente entrenado. La aplicación tiene dos componentes principales:

1. **Interfaz de Usuario con Streamlit**: Permite a los usuarios ingresar los datos de la transacción y obtener una predicción sobre si es fraudulenta o no.
2. **Entrenamiento del Modelo de Detección de Fraude**: El modelo se entrena usando un conjunto de datos de transacciones y se guarda para su posterior uso.

## Requisitos

Para ejecutar este proyecto, asegúrate de tener instalados los siguientes paquetes en tu entorno:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `imblearn`
- `xgboost`
- `pickle`
- `KNNImputer` (de `sklearn.impute`)

Puedes instalar los paquetes necesarios utilizando pip:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn xgboost
```

## Uso

### Modelo en la nube
El modelo se establecio en la nube con la ayuda del servicio Streamlit cloud, solo debes pegar el siguiente link en tu navegador para utilizarlo.

`https://fraud-model-machine-larning-fqpvqityytg8udz4qtcj22.streamlit.app/`

### Interfaz de Usuario con Streamlit

1. Ejecuta el siguiente comando para iniciar la aplicación en Streamlit:

```bash
streamlit run app.py
```

2. Abre el navegador y ve a la dirección proporcionada (por lo general, `http://localhost:8501/`).
3. En la interfaz, selecciona el país y completa los campos correspondientes a las variables de la transacción.
4. Haz clic en el botón "Predecir" para obtener el resultado de la predicción. Si el modelo predice que la transacción es fraudulenta, se mostrará un mensaje de error. Si la predicción es no fraudulenta, se mostrará un mensaje de éxito.

### Entrenamiento del Modelo

El modelo es entrenado usando un conjunto de datos llamado `data.csv` que contiene varias características relacionadas con las transacciones. El flujo para entrenar el modelo incluye:

1. **Preprocesamiento de Datos**:
   - Limpieza de datos (reemplazo de valores faltantes y valores -1 en algunas columnas).
   - Transformación de variables numéricas (aplicación de la raíz cuadrada).
   - One-Hot Encoding de la columna `j` (país).
   
2. **Balanceo de Clases**: Se aplica un undersampling para balancear las clases de fraude (1) y no fraude (0) en el conjunto de entrenamiento.

3. **Entrenamiento del Modelo XGBoost**: El modelo XGBoost se entrena con los datos balanceados.

4. **Guardado del Modelo**: El modelo entrenado se guarda en un archivo `modelo.pkl` para su uso posterior en la aplicación de Streamlit.

### Proceso de Predicción

Cuando el usuario ingresa datos en la aplicación de Streamlit y presiona el botón de predicción, el sistema sigue estos pasos:

1. **Preprocesamiento de Datos de Entrada**:
   - Se aplican las mismas transformaciones que al conjunto de entrenamiento (limpieza, transformación de variables, One-Hot Encoding).
   
2. **Predicción con el Modelo Cargado**:
   - El modelo cargado desde el archivo `modelo.pkl` realiza la predicción de si la transacción es fraudulenta o no.

## Estructura del Proyecto

```plaintext
.
├── app.py             # Aplicación Streamlit para la interfaz de usuario
├── data.csv           # Conjunto de datos de transacciones
├── modelo.pkl         # Modelo XGBoost entrenado
├── requirements.txt   # Dependencias necesarias
├── README.md          # Documentación del proyecto
```

## Contribuciones

Las contribuciones son bienvenidas. Si deseas mejorar este proyecto, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tu mejora o corrección.
3. Realiza los cambios y prueba que funcionan correctamente.
4. Haz un pull request con una descripción clara de los cambios realizados.

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
```

Este README proporciona información clara sobre cómo usar el proyecto, qué dependencias son necesarias, y cómo se estructura el código.