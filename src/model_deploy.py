from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# 1. Inicializar la App
app = FastAPI(
    title="API de Predicción de Riesgo",
    description="Endpoint para predicciones de riesgo crediticio",
    version="1.0.1"
)

# 2. Cargar el modelo y el preprocesador
# Usamos rutas relativas seguras
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_riesgo.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocesador.pkl")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    print(f"❌ Error crítico al cargar los modelos: {e}")

# 3. Esquema de entrada EXACTO
class PatientData(BaseModel):
    creditos_sectorFinanciero: float
    saldo_total: float
    plazo_meses: int
    salario_cliente: float
    creditos_sectorReal: float
    creditos_sectorCooperativo: float
    capital_prestado: float
    huella_consulta: int
    puntaje: float
    puntaje_datacredito: float
    cuota_pactada: float
    saldo_principal: float
    promedio_ingresos_datacredito: float
    tipo_credito: str
    tendencia_ingresos: str
    saldo_mora: float
    edad_cliente: int
    saldo_mora_codeudor: float
    total_otros_prestamos: float
    cant_creditosvigentes: int
    tipo_laboral: str
    antiguedad: int 

class PredictionInput(BaseModel):
    data: list[PatientData]

# 4. Endpoint principal
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([item.dict() for item in input_data.data])
        
        # --- AJUSTE IMPORTANTE ---
        # El preprocesador (Scikit-Learn) suele ser estricto con el orden de las columnas.
        # Nos aseguramos de que el orden sea el mismo que espera el preprocesador.
        # Si el preprocessor fue entrenado con una lista específica, el df debe seguirla.
        # -------------------------
        
        # Aplicar el preprocesador
        data_processed = preprocessor.transform(df)
        
        # Realizar la predicción
        predictions = model.predict(data_processed)
        
        # Opcional: Obtener probabilidades (si el modelo lo permite)
        # probabilities = model.predict_proba(data_processed)[:, 1]
        
        return {
            "status": "success",
            "predictions": predictions.tolist(),
            # "probabilities": probabilities.tolist() 
        }
    except Exception as e:
        # Log del error en la consola del contenedor para debuguear
        print(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. Endpoint de salud
@app.get("/")
def home():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }