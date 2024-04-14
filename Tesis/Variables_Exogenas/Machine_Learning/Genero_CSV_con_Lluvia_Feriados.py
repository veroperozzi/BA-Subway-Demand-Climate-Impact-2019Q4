import pandas as pd

# Cargar el archivo CSV
archivo_csv = "molinetes_concat_linea_A_ordenados_por_fecha.csv"  # Reemplaza con la ruta real de tu archivo CSV
data = pd.read_csv(archivo_csv)

# Seleccionar las columnas requeridas
columnas_requeridas = ["FECHA", "TOTAL", "PRECIPITACION"]
nuevo_data = data[columnas_requeridas]

# Guardar en un nuevo archivo CSV
nuevo_archivo_csv = "Con_Precipitacion.csv"
nuevo_data.to_csv(nuevo_archivo_csv, index=False)

