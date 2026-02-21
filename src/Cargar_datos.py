import os
import pandas as pd

def cargar_datos():
    file_path = r"C:\Users\Enzo\OneDrive\Escritorio\Data Science Henry\Modulo 5\MLOps---DSPTM5\Base_de_datos.xlsx"
    print("Leyendo archivo desde:", file_path)
    df = pd.read_excel(file_path)
    return df




    