import numpy as np
import pandas as pd

def calcular_psi(referencia, actual, buckets=10):
    """Calcula el Population Stability Index"""
    def get_counts(arr, bins):
        return np.histogram(arr, bins=bins)[0] / len(arr)

    # Crear los puntos de corte basados en la data de entrenamiento (referencia)
    breakpoints = np.percentile(referencia, np.arange(0, 101, 100/buckets))
    # Asegurar que los bordes cubran todo
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Obtener proporciones por bucket
    f_ref = get_counts(referencia, bins=breakpoints)
    f_act = get_counts(actual, bins=breakpoints)
    
    # Evitar divisiones por cero
    f_ref = np.where(f_ref == 0, 0.0001, f_ref)
    f_act = np.where(f_act == 0, 0.0001, f_act)
    
    psi = np.sum((f_act - f_ref) * np.log(f_act / f_ref))
    return psi

def evaluar_salud_modelo(y_real, y_pred):
    from sklearn.metrics import f1_score, precision_score
    return {
        "F1": f1_score(y_real, y_pred),
        "Precision": precision_score(y_real, y_pred)
    }