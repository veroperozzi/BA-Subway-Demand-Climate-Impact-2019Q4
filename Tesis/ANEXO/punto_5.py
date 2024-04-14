#Importar librerias
import pandas as pd

# Crear el dataframe de ejemplo
data = {
    'FECHA': ['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2016-01-06'],
    'TOTAL': [13641.0, 46091.0, 24907.0, 162160.0, 171150.0, 174075.0],
    'ESTACION': ['Luro', 'Saens Penia', 'Saens Pena', 'Sanes Penu00d1a', 'Sáenz Peña', 'Mitre']
}

df = pd.DataFrame(data)

# Reemplazar los valores incorrectos en la columna 'ESTACION'
correcciones = {
    'Saens Penia': 'Sáenz Peña',
    'Saens Pena': 'Sáenz Peña',
    'Sanes Penu00d1a': 'Sáenz Peña'
}

df['ESTACION'] = df['ESTACION'].replace(correcciones)

# Guardar el nuevo dataframe en un archivo CSV
df.to_csv("linea_A_estacion_modificada.csv", index=False)
