import pandas as pd
import numpy as np

filname = 'Con_Precipitacion_Feriado_Laborable.csv'

#Leer CSV e imprimo las primeras 5 lineas
df = pd.read_csv(filname)
print(df.head())

df = pd.DataFrame(df)

# Calcular la matriz de correlación
correlation_matrix = df.drop("FECHA", axis=1).corr()

# Imprimir la matriz de correlación
print(correlation_matrix)
