#Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Lee archivo CSV
filname = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filname)

# Selecciona la columna de interés
serie_a_analizar = df['TOTAL']


def hampel_filter(data, m, k):
    rolling_median = data.rolling(window=m).median()
    mad = np.abs(data - rolling_median)
    mad_median = mad.rolling(window=m).median()
    threshold = k * mad_median
    is_outlier = mad > threshold
    data_filtered = data.copy()
    data_filtered[is_outlier] = rolling_median[is_outlier]
    return data_filtered

m = 365  # Tamaño de la ventana
k = 3   # Factor de umbral

data_filtered = hampel_filter(df['TOTAL'], m, k)

# Convertir la serie filtrada en una lista
resultados_lista = data_filtered.tolist()

# Imprimir o trabajar con la lista de resultados
print(resultados_lista)

#Grafico
plt.figure(figsize=(12, 6))
plt.plot(df['TOTAL'], label='Serie Original')
plt.plot(data_filtered, label='Serie Filtrada', color='red')
plt.legend()
plt.title('Comparación entre la Serie Original y la Serie Filtrada')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.show()