import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def limpiar_datos(ruta_archivo_entrada, ruta_archivo_salida):
    # Cargar los datos
    print(f"Cargando datos desde {ruta_archivo_entrada}...")
    df = pd.read_csv(ruta_archivo_entrada)
    
    # 1. Eliminar 'id', 'diagnosis' y 'Unnamed: 32' (columna residual común en este dataset)
    columnas_a_eliminar = ['id', 'diagnosis', 'diagnostic', 'Unnamed: 32']
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    df_limpio = df.drop(columns=columnas_existentes)
    print(f"Se eliminaron las columnas: {columnas_existentes}")
    
    # 2. Normalizar todas las columnas
    print("Normalizando las columnas con StandardScaler (media=0, varianza=1)...")
    scaler = StandardScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_limpio), columns=df_limpio.columns)
    
    # 3. Estudiar la correlación entre cada variable
    print("Calculando correlaciones...")
    # Usamos .abs() para identificar tanto > 0.95 como < -0.95
    corr_matrix = df_norm.corr().abs()
    
    # Seleccionamos la parte triangular superior de la matriz de correlación
    # para no eliminar ambas variables correlacionadas, sino solo una de ellas.
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 4. Buscamos variables con una correlación mayor al 95%
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    print(f"Se eliminarán {len(to_drop)} columnas con una correlación muy alta (> 0.95 o < -0.95):")
    print(to_drop)
    
    # Quedarse solo con las variables no redundantes
    df_final = df_norm.drop(columns=to_drop)
    
    print(f"\nEl dataset ha pasado de {df_limpio.shape[1]} a {df_final.shape[1]} columnas.")
    
    # Guardamos el archivo
    # IMPORTANTE: Se guarda fuera de "Bases de datos" para respetar las reglas.
    df_final.to_csv(ruta_archivo_salida, index=False)
    print(f"Datos limpios guardados en: {ruta_archivo_salida}")
    
    return df_final

if __name__ == '__main__':
    archivo_entrada = "Bases de datos/BreastCancer.csv"
    archivo_salida = "BreastCancer_limpio.csv"
    limpiar_datos(archivo_entrada, archivo_salida)
