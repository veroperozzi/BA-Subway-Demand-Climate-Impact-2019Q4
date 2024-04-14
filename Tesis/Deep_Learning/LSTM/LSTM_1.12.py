#Importacion de librerias y configuracion
import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

#Carga y exploracion de datos
filename = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filename)
print(df)

# Imprime primeras filas
print(df.head())
# Imprime ultimas filas
print(df.tail())

#Preprocesamiento de datos
# Convierte columna fecha en valores Datatime
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Imprime total columnas y filas
print(df.shape)
# Elimina duplicados en base a columna Fecha
df = df.drop_duplicates(subset='FECHA', ignore_index=True)
# Imprime total columnas y filas
print(df.shape)

#Visualizacion de los datos
# Grafico serie temporal completa
fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(df.TOTAL)
ax.set_xlabel('Periodo')
ax.set_ylabel('Volumen Pasajeros')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Grafico de un mes
fig, ax = plt.subplots()
ax.plot(df['TOTAL'],'b-',)
ax.set_xlabel('Periodo')
ax.set_ylabel('Volumen Pasajeros')
plt.xticks(np.arange(0, 30, 1), ['Friday', 'Saturday', 'Sunday',
                                 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                                 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                                 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                                 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.xlim(0, 30)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

#Analisis especifico por anio
# Funcion para graficar un anio de la serie zoomeado
def zoom_por_anio(anio):
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    # Filtrar datos solo para el año 2018 y 2019
    df_anio = df[(df['FECHA'].dt.year == 2018) | (df['FECHA'].dt.year == 2019)]

    # Graficar serie temporal de los años 2018 y 2019
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df_anio['FECHA'], df_anio['TOTAL'])
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Volumen Pasajeros')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# zoom_por_anio(2018)

# esto es una funcion para graficar dos anio de la serie zoomeado
def zoom_plot_years(anio_inicio, anio_final):
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    # Filtrar datos solo para el año 2018 y 2019
    df_anio = df[(df['FECHA'].dt.year == anio_inicio) | (df['FECHA'].dt.year == anio_final)]

    # Graficar serie temporal de los años 2018 y 2019
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df_anio['FECHA'], df_anio['TOTAL'])
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Volumen Pasajeros')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


zoom_plot_years(2018, 2019)

# FEATURE ENGUNEERING y divicion de datos

print('CARACTERISTICAS DE LOS DATOS')
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
print(df.describe().transpose())

# Codificar tiempo como caracteristica utilizable para el modelo de aprendizaje profundo.
# El modelo no puede usar la f(x) date_time, ya que es una cadena de fecha y hora.
# Expresamos fecha como numero de segundos, mediante metodo de marca de timepo.
print('CODIFICAMOS TIEMPO')
timestamp_s = pd.to_datetime(df['FECHA']).map(datetime.datetime.timestamp)
print(timestamp_s)
# "timestamp_s" es una nueva serie que contiene los valores de timestamp en SEGUNDOS para c/fecha de la
# columna FECHA

# Grafico
# Se utiliza la transformacion de Fourier para analizar una señal de datos.
# se tienen datos numéricos que representan una cantidad (en este caso, el total de pasajeros en un
# determinado día). La transformada de Fourier se aplica a esos datos para analizar las diferentes
# frecuencias presentes en ellos (se otorga un peso/ presencia de esos datos). Las frecuencias
# se refieren a los diferentes patrones repetitivos o ciclos en los datos.
# Fourier es una herramienta matemática que permite descomponer una señal en sus diferentes componentes
# de frecuencia, lo cual puede ser útil para analizar patrones o características específicas en los datos.
fft = tf.signal.rfft(df['TOTAL'])
f_per_dataset = np.arange(0, len(fft))

n_sample_h = len(df['TOTAL'])
hours_per_week = 24 * 7
weeks_per_dataset = n_sample_h / hours_per_week

f_per_week = f_per_dataset / weeks_per_dataset

plt.step(f_per_week, np.abs(fft))
plt.xscale('log')
plt.xticks([1, 7], ['1/week', '1/day'])
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()

# Recuperar el comportamiento ciclico del tiempo
# Calcular el día del año
df['day_of_year'] = df['FECHA'].dt.dayofyear

# Calcular las características ciclicas
day_of_year = df['day_of_year']
df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)

# Eliminar las columnas innecesarias
df = df.drop(['FECHA', 'day_of_year'], axis=1)

# Graficar los puntos en forma de círculo
df.sample(50).plot.scatter('day_sin', 'day_cos').set_aspect('equal')
plt.tight_layout()
plt.show()

# Gráfico que muestra la evolución de los valores de timestamp en segundos a lo largo del tiempo.
fig, ax = plt.subplots()

ax.plot(timestamp_s)
ax.set_xlabel('Time')
ax.set_ylabel('Number of seconds')

plt.xticks([0, len(timestamp_s) - 1], [2016, 2019])

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Dividimos el dataset en test y train
n = len(df)

train_df = df[0:1366]
test_df = df[1366:1457]
# Imprimir dimenciones
print(train_df.shape)
print(test_df.shape)

# #Se escalan valores para que esten entre 0 y 1. Reduce tiempo requerido para entrenar modelo y mejora
# #rendimiento. Colocamos escalador en el conjunto de entrenaineto para evitar fuga de datos.
scaler = MinMaxScaler()
scaler.fit(train_df)

train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

train_df.describe().transpose()

# Obtenemos la ruta actual del script
import os

current_dir = os.getcwd()
print(current_dir)
#Guardar los conjuntos de datos
train_df.to_csv('/home/vero/Documentos/Tesis/Deep_Learning/LSTM/train.csv')
test_df.to_csv('/home/vero/Documentos/Tesis/Deep_Learning/LSTM/test.csv')

print(test_df)

