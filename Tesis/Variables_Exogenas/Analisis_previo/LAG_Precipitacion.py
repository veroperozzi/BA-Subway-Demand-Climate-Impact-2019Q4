import pandas as pd

# Cargar los datos originales
filename = 'Con_Precipitacion.csv'
df = pd.read_csv(filename)

# Crear nuevos retrasos en la variable exógena
df['PRECIPITACION_LAG1'] = df['PRECIPITACION'].shift(1)
df['PRECIPITACION_LAG2'] = df['PRECIPITACION'].shift(2)

# Eliminar filas con valores faltantes después de crear los retrasos
# df = df.dropna()

# Guardar el DataFrame con los retrasos en un nuevo archivo CSV
new_filename = 'Con_Precipitacion_Retrasos.csv'
df.to_csv(new_filename, index=False)

