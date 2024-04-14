import pandas as pd

# Cargar los datos originales
filename = 'Con_Laborable.csv'
df = pd.read_csv(filename)

# Crear nuevos retrasos en la variable exógena
df['LABORABLE_LAG1'] = df['LABORABLE'].shift(1)
df['LABORABLE_LAG2'] = df['LABORABLE'].shift(2)

# Eliminar filas con valores faltantes después de crear los retrasos
# df = df.dropna()

# Guardar el DataFrame con los retrasos en un nuevo archivo CSV
new_filename = 'Con_Laborable_Retrasos.csv'
df.to_csv(new_filename, index=False)

