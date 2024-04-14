#Importar librerias
import pandas as pd

# Intenta leer el archivo con una codificación específica para manejar caracteres especiales

file = 'molinetes_2016.csv'
try:
    df = pd.read_csv(file, encoding='ISO-8859-1')
except UnicodeDecodeError:
    df = pd.read_csv('datos.csv', encoding='cp1252')

df['ESTACION'] = df['ESTACION'].astype(str)
df['LINEA'] = df['LINEA'].astype(str)

# Agrupa por 'LINEA' y cuenta las ESTACIONes únicas por línea
LINEA_ESTACIONes = df.groupby('LINEA')['ESTACION'].nunique()

# Agrupando por "LINEA" y contando las ESTACIONes únicas para cada línea
ESTACIONes_por_LINEA = df.groupby("LINEA")["ESTACION"].nunique().reset_index(name='Cantidad de ESTACIONes')

# Encuentra la línea con el mayor número de ESTACIONes únicas
LINEA_mas_ESTACIONes = ESTACIONes_por_LINEA.loc[ESTACIONes_por_LINEA['Cantidad de ESTACIONes'].idxmax()]

# Encuentra la línea con el menor número de ESTACIONes únicas
LINEA_menos_ESTACIONes = ESTACIONes_por_LINEA.loc[ESTACIONes_por_LINEA['Cantidad de ESTACIONes'].idxmin()]

# Imprimiendo los resultados
print("Cantidad de ESTACIONes por línea:")
print(ESTACIONes_por_LINEA)
print("\nLínea con más ESTACIONes:")
print(LINEA_mas_ESTACIONes)
print("\nLínea con menos ESTACIONes:")
print(LINEA_menos_ESTACIONes)



# Calcular la suma de TOTAL por cada LINEA
TOTAL_por_LINEA = df.groupby('LINEA')['TOTAL'].sum().reset_index()

# Encontrar la LINEA con el valor máximo de TOTAL
LINEA_max_TOTAL = TOTAL_por_LINEA.loc[TOTAL_por_LINEA['TOTAL'].idxmax()]

print("TOTAL por cada línea:")
print(TOTAL_por_LINEA)
print("\nLínea con el valor máximo de TOTAL:")
print(LINEA_max_TOTAL)


