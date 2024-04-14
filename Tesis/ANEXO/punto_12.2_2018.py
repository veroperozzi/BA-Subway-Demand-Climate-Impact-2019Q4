import pandas as pd

# Cargar el CSV
df = pd.read_csv('molinetes_2018_con_feriados.csv')

# Transformar los nombres de las columnas a mayúsculas
df.columns = [column.upper() for column in df.columns]
print(df['ESTACION'].unique())
df['ESTACION'] = df['ESTACION'].replace('Saenz Peña ', 'SAENZ PENIA')
df['ESTACION'] = df['ESTACION'].replace('Agüero', 'AGUERO')

df['ESTACION'] = df['ESTACION'].str.upper()

# Obtiene los valores únicos de la columna 'ESTACION'
estaciones_unicas = df['ESTACION'].unique()

# Cuenta el número de estaciones únicas
numero_de_estaciones_unicas = len(estaciones_unicas)
print(numero_de_estaciones_unicas)

# Diccionarios con las claves en mayúsculas

Residencial = {
    'PASCO': None,
    'ALBERTI': None,
    'PLAZA MISERERE': None,
    'LORIA': None,
    'CASTRO BARROS': None,
    'RIO DE JANEIRO': None,
    'ACOYTE': None,
    'PRIMERA JUNTA': None,
    'PUAN': None,
    'CARABOBO': None,
    'FLORES': None,
    'SAN PEDRITO': None,
    'PASTEUR': None,
    'PUEYRREDON': None,
    'PUEYRREDON.D': None,
    'CARLOS GARDEL': None,
    'MEDRANO': None,
    'ANGEL GALLARDO': None,
    'MALABIA': None,
    'DORREGO': None,
    'FEDERICO LACROZE': None,
    'TRONADOR': None,
    'LOS INCAS': None,
    'ECHEVERRIA': None,
    'ROSAS': None,
    'AGUERO': None,
    'BULNES': None,
    'SCALABRINI ORTIZ': None,
    'PLAZA ITALIA': None,
    'PALERMO': None,
    'MINISTRO CARRANZA': None,
    'OLLEROS': None,
    'JOSE HERNANDEZ': None,
    'JURAMENTO': None,
    'CONGRESO DE TUCUMAN': None,
    'PICHINCHA': None,
    'JUJUY': None,
    'URQUIZA': None,
    'BOEDO': None,
    'AVENIDA LA PLATA': None,
    'JOSE MARIA MORENO': None,
    'EMILIO MITRE': None,
    'MEDALLA MILAGROSA': None,
    'VARELA': None,
    'PZA. DE LOS VIRREYES': None,
    'CORDOBA': None,
    'CORRIENTES': None,
    'ONCE': None,
    'VENEZUELA': None,
    'HUMBERTO I': None,
    'INCLAN': None,
    'CASEROS': None,
    'PATRICIOS': None,
    'HOSPITALES': None,
}

Centrico = {
    'PLAZA DE MAYO': None,
    'PERU': None,
    'PIEDRAS': None,
    'LIMA': None,
    'SAENZ PENIA': None,
    'CONGRESO': None,
    'LEANDRO N. ALEM': None,
    'FLORIDA': None,
    'CARLOS PELLEGRINI': None,
    'URUGUAY': None,
    'CALLAO': None,
    'CALLAO.B': None,
    'RETIRO': None,
    'GENERAL SAN MARTIN': None,
    'LAVALLE': None,
    'DIAGONAL NORTE': None,
    'AVENIDA DE MAYO': None,
    'MARIANO MORENO': None,
    'INDEPENDENCIA': None,
    'INDEPENDENCIA.H': None,
    'SAN JUAN': None,
    'CONSTITUCION': None,
    'CATEDRAL': None,
    '9 DE JULIO': None,
    'TRIBUNALES': None,
    'FACULTAD DE MEDICINA': None,
    'PUEYRREDON': None,
    'FACULTAD DE DERECHO': None,
    'LAS HERAS': None,
    'SANTA FE': None,
    'BOLIVAR': None,
    'GENERAL BELGRANO': None,
    'SAN JOSE': None,
    'ENTRE RIOS': None,
}

ZONA = []

for index, row in df.iterrows():
    nombre_estacion = row['ESTACION']
    if nombre_estacion in Residencial:
        ZONA.append(1)
    elif nombre_estacion in Centrico:
        ZONA.append(0)
    else:
        ZONA.append(2)

df['ZONA'] = ZONA

# Que el formato original de la fecha coincida con el argumento de `format`
df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y')

# Ahora reformateamos la fecha al nuevo formato
df['FECHA'] = df['FECHA'].dt.strftime('%Y-%m-%d')

#Convertir columna en float
df['PRECIPITACION'] = pd.to_numeric(df['PRECIPITACION'], errors='coerce')

df['LINEA'] = df['LINEA'].str.replace('Linea', '')

print(df.head())
df.to_csv('molinetes_2018_con_zonas.csv', index=False)
