# 1. Usar una imagen base ligera de Python
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar el archivo de requerimientos e instalar librerías
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar la carpeta src y los modelos
# (Docker copiará todo lo que esté en tu carpeta local src a /app/src)
COPY src/ ./src/

# 6. Exponer el puerto que usará FastAPI
EXPOSE 8000

# 7. Comando para ejecutar la API al iniciar el contenedor
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]