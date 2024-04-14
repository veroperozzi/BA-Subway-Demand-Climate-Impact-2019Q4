#Importacion de librerias y configuracion
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

#Carga y exploracion de datos
filename = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filename)
print(df)

df['FECHA'] = pd.to_datetime(df['FECHA'])
df.info()

# Separar datos de entrenamiento y prueba
train = df.iloc[0:1366]
test_df = df.iloc[1366:1457]

# Calculamos el mínimo y máximo de los datos combinados
min_data = df['TOTAL'].min()
max_data = df['TOTAL'].max()

# MAE normalizado proporcionado
mae_normalizado = 0.045

# Calculamos el rango de los datos
rango_datos = max_data - min_data

# Desnormalizamos el MAE
mae_desnormalizado = mae_normalizado * rango_datos

print(min_data, max_data, mae_desnormalizado)

