import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Determino los 3 meses de cada anio con mayor demanda de la linea A del subte de Bs.As.
# Crear un DataFrame de pandas a partir de los datos proporcionados
df = pd.read_csv('linea_A_concat_xdia_total.csv')

# Agregar columnas de año y mes para facilitar el agrupamiento
tipo_dato = df['FECHA'].dtype
print(tipo_dato)
df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
df['AÑO'] = df['FECHA'].dt.year
df['MES'] = df['FECHA'].dt.month
print(df)
# Agrupo por anio y mes y sumo los totales
total_x_mes = df.groupby(['AÑO', 'MES'])['TOTAL'].sum().reset_index()
total_x_mes['TOTAL'].astype(int)
print(total_x_mes)

# Identificamos los 3 meses con mayor total de pasajeros por año
top_meses_x_anio = total_x_mes.groupby('AÑO').apply(lambda x: x.nlargest(3, 'TOTAL')).reset_index(drop=True)

print(top_meses_x_anio)

####################################

df_1 = pd.read_csv('Con_Precipitacion_Feriado_Laborable.csv')
print(df_1)

# Renombro columna Total como Demanda de Pasajeros
df_1.rename(columns={'TOTAL': 'Dem_Pasaj'}, inplace=True)

df_1['Cap_Transp'] = df_1.apply(lambda row: 123000 if row['FERIADO'] == 1 and
                                                      row['LABORABLE'] == 1 else (360000 if row['FERIADO'] == 0 and
                                                        row['LABORABLE'] == 1 else 123000 if row['LABORABLE'] == 0 else 0), axis=1)
df_1.drop(columns=['LABORABLE'], inplace=True)

df_1['Pct_Cap_Util'] = round(((df_1['Dem_Pasaj'] / df_1['Cap_Transp']) * 100), 2)
df_1['Pct_Cap_Ociosa'] =  round((100 - df_1['Pct_Cap_Util']), 2)

df_1.info()
df_1.to_csv('capacidad_subterraneo.csv', index=False)
print(df_1)

################################################
# Grafico AGOSTO 2016
print(plt.style.available)

df_1['FECHA'] = pd.to_datetime(df_1['FECHA'])
# Filtrar por los meses de interés
df_filtered = df_1[(df_1['FECHA'] >= '2016-08-01') & (df_1['FECHA'] <= '2016-08-31')]
# Establecer el estilo de los gráficos
plt.style.use('seaborn-v0_8-darkgrid')
# gráfico de líneas para la demanda de pasajeros y la capacidad de transporte
plt.figure(figsize=(8, 4))
plt.plot(df_filtered['FECHA'], df_filtered['Dem_Pasaj'], label='Demanda de Pasajeros', color='blue')
# Añadir línea horizontal para la capacidad de transporte en Semana
plt.axhline(y=360000, color='green', linestyle='-', linewidth=2, label='Capacidad de Transporte en Semana')
# Añadir línea horizontal para la capacidad de transporte en Fin de semana y Feriados
plt.axhline(y=126000, color='orange', linestyle='-', linewidth=2, label='Capacidad de Transporte en Fin de Semana/Feriados')
# título y leyenda al gráfico
plt.title('Demanda de Pasajeros vs. Capacidad de Transporte (Ago 2016)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Pasajeros')
plt.legend()

# Mejorar la disposición de las fechas en el eje x
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Grafico AGOSTO 2017
print(plt.style.available)

df_1['FECHA'] = pd.to_datetime(df_1['FECHA'])
# Filtrar por los meses de interés
df_filtered = df_1[(df_1['FECHA'] >= '2017-08-01') & (df_1['FECHA'] <= '2017-08-31')]
# Establecer el estilo de los gráficos
plt.style.use('seaborn-v0_8-darkgrid')
# gráfico de líneas para la demanda de pasajeros y la capacidad de transporte
plt.figure(figsize=(8, 4))
plt.plot(df_filtered['FECHA'], df_filtered['Dem_Pasaj'], label='Demanda de Pasajeros', color='blue')
# Añadir línea horizontal para la capacidad de transporte en Semana
plt.axhline(y=360000, color='green', linestyle='-', linewidth=2, label='Capacidad de Transporte en Semana')
# Añadir línea horizontal para la capacidad de transporte en Fin de semana y Feriados
plt.axhline(y=126000, color='orange', linestyle='-', linewidth=2, label='Capacidad de Transporte en Fin de Semana/Feriados')
# título y leyenda al gráfico
plt.title('Demanda de Pasajeros vs. Capacidad de Transporte (Ago 2017)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Pasajeros')
plt.legend()
# Mejorar la disposición de las fechas en el eje x
plt.xticks(rotation=45)
plt.tight_layout()
# Mostrar el gráfico
plt.show()

# Grafico AGOSTO 2018
print(plt.style.available)

df_1['FECHA'] = pd.to_datetime(df_1['FECHA'])
# Filtrar por los meses de interés
df_filtered = df_1[(df_1['FECHA'] >= '2018-08-01') & (df_1['FECHA'] <= '2018-08-31')]
# Establecer el estilo de los gráficos
plt.style.use('seaborn-v0_8-darkgrid')
# gráfico de líneas para la demanda de pasajeros y la capacidad de transporte
plt.figure(figsize=(8, 4))
plt.plot(df_filtered['FECHA'], df_filtered['Dem_Pasaj'], label='Demanda de Pasajeros', color='blue')
# Añadir línea horizontal para la capacidad de transporte en Semana
plt.axhline(y=360000, color='green', linestyle='-', linewidth=2, label='Capacidad de Transporte en Semana')
# Añadir línea horizontal para la capacidad de transporte en Fin de semana y Feriados
plt.axhline(y=126000, color='orange', linestyle='-', linewidth=2, label='Capacidad de Transporte en Fin de Semana/Feriados')
# título y leyenda al gráfico
plt.title('Demanda de Pasajeros vs. Capacidad de Transporte (Ago 2018)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Pasajeros')
plt.legend()

# Mejorar la disposición de las fechas en el eje x
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Grafico Octubre 2019
print(plt.style.available)

df_1['FECHA'] = pd.to_datetime(df_1['FECHA'])
# Filtrar por los meses de interés
df_filtered = df_1[(df_1['FECHA'] >= '2019-10-01') & (df_1['FECHA'] <= '2019-10-31')]
# Establecer el estilo de los gráficos
plt.style.use('seaborn-v0_8-darkgrid')
# gráfico de líneas para la demanda de pasajeros y la capacidad de transporte
plt.figure(figsize=(8, 4))
plt.plot(df_filtered['FECHA'], df_filtered['Dem_Pasaj'], label='Demanda de Pasajeros', color='blue')
# Añadir línea horizontal para la capacidad de transporte en Semana
plt.axhline(y=360000, color='green', linestyle='-', linewidth=2, label='Capacidad de Transporte en Semana')
# Añadir línea horizontal para la capacidad de transporte en Fin de semana y Feriados
plt.axhline(y=126000, color='orange', linestyle='-', linewidth=2, label='Capacidad de Transporte en Fin de Semana/Feriados')
# título y leyenda al gráfico
plt.title('Demanda de Pasajeros vs. Capacidad de Transporte (Oct 2019)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Pasajeros')
plt.legend()

# Mejorar la disposición de las fechas en el eje x
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()