import pandas as pd

# Leemos los datos del CSV
df = pd.read_csv('Con_Precio.csv')
# Convertimos la columna 'FECHA' al formato deseado
df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# Guardamos el DataFrame corregido en un nuevo archivo CSV
df.to_csv('fechas_corregidas.csv', index=False)

df.head()
