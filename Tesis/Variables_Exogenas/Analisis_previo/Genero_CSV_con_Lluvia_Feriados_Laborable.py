import pandas as pd

# Cargar el archivo CSV
archivo_csv = "Con_Precipitacion_Retrasos.csv"
data = pd.read_csv(archivo_csv)

# Seleccionar las columnas requeridas
columnas_requeridas = ["FECHA", "TOTAL","PRECIPITACION_LAG2"]
nuevo_data = data[columnas_requeridas]

# Guardar en un nuevo archivo CSV
nuevo_archivo_csv = "Con_Precipitacion_LAG2.csv"
nuevo_data.to_csv(nuevo_archivo_csv, index=False)

