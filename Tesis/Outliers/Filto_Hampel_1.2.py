#Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcular_MAD(datos):
    mediana = np.median(datos)
    desviaciones_absolutas = np.abs(datos - mediana)
    MAD = np.median(desviaciones_absolutas)
    return MAD


def identificar_valores_atipicos(datos, ventana, k):
    """
    Identifica valores atípicos en una serie temporal usando el filtro de Hampel.

    :param datos: Array de datos numéricos (serie temporal).
    :param ventana: Tamaño de la ventana para calcular la mediana y la MAD.
    :param k: Factor multiplicativo para determinar el umbral de detección de valores atípicos.
    :return: Indices de los valores atípicos.
    """
    valores_atipicos = []
    n = len(datos)
    for i in range(n):
        inicio = max(0, i - ventana)
        fin = min(n, i + ventana + 1)

        mediana = np.median(datos[inicio:fin])
        MAD = calcular_MAD(datos[inicio:fin])

        umbral = k * MAD

        if abs(datos[i] - mediana) > umbral:
            valores_atipicos.append(i)

    return valores_atipicos


# Lee tus datos desde el archivo CSV
filename = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filename)

# Selecciona la columna de interés (en este caso, 'TOTAL')
serie = df['TOTAL']

# Parámetros del filtro de Hampel
ventana = 365
k = 3


# Calcular límites para identificar valores atípicos
limite_inferior = []
limite_superior = []
for i in range(len(serie)):
    inicio = max(0, i - ventana)
    fin = min(len(serie), i + ventana + 1)

    mediana = np.median(serie[inicio:fin])
    MAD = calcular_MAD(serie[inicio:fin])

    limite = k * MAD
    limite_inferior.append(mediana - limite)
    limite_superior.append(mediana + limite)



# Identificar valores atípicos
indices_valores_atipicos = identificar_valores_atipicos(serie, ventana, k)

print("Índices de valores atípicos:", indices_valores_atipicos)

# Gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(range(len(serie)), serie, label='Datos', color='blue')
plt.scatter(indices_valores_atipicos, serie.iloc[indices_valores_atipicos], color='red', label='Valores Atípicos')
plt.plot(limite_inferior, label='Límite Inferior', color='green', linestyle='--')
plt.plot(limite_superior, label='Límite Superior', color='orange', linestyle='--')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
plt.xlabel('Periodo')
plt.ylabel('Pasajeros linea A')
plt.title('Gráfico de Dispersión con Valores Atípicos')
plt.legend(loc='upper right')
plt.show()
