import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from Cargar_datos import cargar_datos  

TARGETS = "Pago_atiempo"

COLUMNA_NUMERICAS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "puntaje",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]

COLUMNAS_CATEGORICAS = ["tipo_laboral", "tendencia_ingresos"]
COLUMNAS_ORDINALES = ["tipo_credito"] 
EXCLUIR_DE_X = ["fecha_prestamo"]

def preparar_datos(df):
 
    for col in COLUMNAS_CATEGORICAS + COLUMNAS_ORDINALES:
        df[col] = df[col].astype(str)


    X = df.drop(columns=[TARGETS] + EXCLUIR_DE_X)
    y = df[TARGETS]

  
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


    all_categorical = COLUMNAS_CATEGORICAS + COLUMNAS_ORDINALES
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])


    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, COLUMNA_NUMERICAS),
        ("cat", categorical_pipeline, all_categorical)
    ])


    X_transformed = preprocessor.fit_transform(X)

    return X_transformed, y, preprocessor

def split_datos(df, test_size=0.2, random_state=42):

    X, y, preprocessor = preparar_datos(df) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor       

    
