import os
import pandas as pd

def cargar_datos():
    "Funcion para Cargar Datos desde un Archivo de Excel"

    script_dir = os.path.dirname(__file__)  # Directorio del script actual

    file_path = os.path.join(script_dir, 'Base_de_datos.xlsx')

    df = pd.read_excel(file_path)

    return df
