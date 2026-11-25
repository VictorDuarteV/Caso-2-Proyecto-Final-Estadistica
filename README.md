# Predicción del valor de propiedades inmobiliarias (Caso 2, EAI)

Proyecto Final de Estadística Aplicada a la Investigación.
Alumno: Victor Francisco Duarte Villalobos (403230). Maestría en Ingeniería en Computación, UACH.

Dashboard en Streamlit con 7 secciones: Proyecto, Datos (limpieza, diccionario, descriptivos, mapa,
precios históricos, filtros y descarga), Exploración (distribuciones, Pearson/Spearman, ANOVA),
Regresión (regresión lineal múltiple con valores-p), Modelos (CV, Test, real vs predicho, error por
rango, importancia de variables), Predicción y Conclusiones.

## Archivos

```
app.py                         Aplicación
.streamlit/config.toml         Tema (colores y Montserrat)
datos/datos_casas_limpio.csv   Dataset limpio (21,596 ventas de King County, 2014-2015)
modelos/*.joblib               Gradient Boosting, Random Forest y Regresión lineal (Pipelines completos)
modelos/resultados.json        Métricas que genera el notebook
modelos/predicciones_test.csv  Predicciones del conjunto de prueba
requirements.txt               Versiones exactas con las que se guardaron los modelos
```

## Ejecución local (Python 3.13)

```
pip install -r requirements.txt
streamlit run app.py
```
