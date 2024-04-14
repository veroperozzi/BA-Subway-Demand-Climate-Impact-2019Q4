import pandas as pd

# Tu conjunto de datos
filname = 'molinetes_concat_linea_A_ordenados_por_fecha.csv'

# Crear un DataFrame a partir de los datos
df = pd.read_csv(filname)

# Mapear los días de la semana a la columna LABORABLE
df['LABORABLE'] = df['DIA DE LA SEMANA'].apply(lambda x: 1 if x in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 0)

# Guardar el DataFrame resultante en un nuevo archivo CSV
df.to_csv('linea_A_con_laborable', index=False)
