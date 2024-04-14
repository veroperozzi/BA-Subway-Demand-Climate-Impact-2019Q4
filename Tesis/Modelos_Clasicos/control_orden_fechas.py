import pandas as pd

# Cambiar nombre CSV de entrada y salida
input_filename = "linea_A_concat_xdia_total.csv"

# Leer el archivo CSV en un pandas DataFrame
df = pd.read_csv(input_filename)

# Convertir la columna FECHA en formato datetime
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Verificar si la primera fecha es menor que la segunda fecha
if df.iloc[0]['FECHA'] > df.iloc[1]['FECHA']:
    print(f"ERROR: La primera fecha ({df.iloc[0]['FECHA']}) es mayor que la segunda fecha ({df.iloc[1]['FECHA']}).")
else:
    # Verificar si la columna FECHA está ordenada de manera ascendente
    if not df['FECHA'].is_monotonic_increasing:
        df_sorted = df.sort_values('FECHA')
        # Encontrar los índices donde las fechas no están ordenadas de manera ascendente
        idx = df_sorted.index[df_sorted['FECHA'].diff() < pd.Timedelta(0)].tolist()
        # Imprimir los registros donde las fechas no están ordenadas de manera ascendente
        print(f"ERROR: Las fechas no están ordenadas de manera ascendente en los siguientes registros:")
        for i in idx:
            fecha_incorrecta = df_sorted.loc[i, 'FECHA']
            print(f"\tFecha incorrecta: {fecha_incorrecta}")
    else:
        print("Las fechas están ordenadas de manera ascendente.")


print(df)