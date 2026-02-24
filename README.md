MLOPS - DSPTM5: Sistema de Predicción de Riesgo

Este repositorio contiene el ciclo de vida completo de un modelo de Machine Learning, desde la ingesta y exploración de datos hasta su productivización mediante una API y un Dashboard, todo contenedorizado con Docker

📋 Resumen del Caso de Negocio
El objetivo es construir un sistema robusto que permita identificar el nivel de riesgo de usuarios. El proyecto utiliza un enfoque de Integración Continua y Despliegue Continuo (CI/CD) basado en versiones (V1.0.0 a V1.1.0), asegurando la trazabilidad de cada experimento.

🛠️ Stack Tecnológico
Lenguaje: Python 3.x

Análisis de Datos: Pandas, Numpy, Matplotlib, Seaborn.

ML & Pipelines: Scikit-Learn (Pipelines, Transformers).

Monitoreo: KS Test, PSI, Jensen-Shannon Divergence.

Interfaz: Streamlit (Dashboard de monitoreo).

Despliegue: FastAPI/Flask, Docker, Docker Compose.

📂 Estructura y Avances del Proyecto
1. Exploración y Versionamiento (V1.0.1)
Cargar_datos.py: Simula la extracción de un DWH utilizando archivos locales .csv y .xlsx.

Comprension_eda.ipynb:

Caracterización: Identificación de variables categóricas, numéricas y temporales.

Limpieza: Unificación de nulos y corrección de tipos de datos.

Análisis: Gráficos univariables (histogramas, boxplots), bivariables (vs Target) y multivariables (matrices de correlación).

2. Ingeniería y Modelamiento (V1.1.0)
ft_engineering.py: Pipeline de transformación de datos. Genera los conjuntos de entrenamiento y evaluación.

model_training_evaluation.ipynb: Entrenamiento de modelos supervisados.

Uso de funciones build_model y summarize_classification.

Comparativa de modelos mediante tablas de métricas (Accuracy, F1-Score, AUC).

Exportación de artefactos: modelo_riesgo.pkl y preprocesador.pkl.

3. Monitoreo y detección de Drift
model_monitoring.py: Sistema de control de calidad del modelo en producción.

Muestreo: Análisis periódico de la data entrante vs data de entrenamiento.

Métricas de Drift: Implementación de pruebas Kolmogorov-Smirnov (KS) y PSI para detectar cambios en la distribución de la población.

Dashboard (Streamlit): Visualización interactiva de alertas y análisis temporal del desempeño.

4. Disponibilización y Dockerización
model_deploy.py: API que expone el endpoint /predict. Soporta inferencia por lotes (batch processing).

Dockerización: * Dockerfile.api: Contenedor para el servicio de predicción.

Dockerfile.dashboard: Contenedor para la interfaz de Streamlit.

docker-compose.yml: Orquestación completa de los servicios.

🚀 Guía de Inicio Rápido
Configuración del Entorno
Clonar el repositorio y acceder a la rama developer.

Crear entorno virtual: python -m venv .venv

Instalar dependencias: pip install -r requirements.txt

📈 Hallazgos Principales

Observando los resultados, nos encontramos con un escenario de "Modelos Perfectos". En el mundo real del ML, un Accuracy o F1-score de 1.000000 suele ser una señal de alerta (posible Data Leakage o sobreajuste extremo), pero para fines del proyecto integrador, demuestra que el pipeline de ingeniería de atributos fue sumamente efectivo.
