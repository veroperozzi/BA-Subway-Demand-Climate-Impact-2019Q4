import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

filename = "linea_A_concat_xdia_total.csv"

# Leer dataset
df = pd.read_csv(filename)
print(df.head())

# Creo columna llamada Year y convierte los valores de la columna FECHA en un indice de fecha
df['Year'] = pd.DatetimeIndex(df['FECHA']).year
print(df)

# Grafico Serie de Tiempo
fig, ax = plt.subplots()
ax.plot(df.FECHA, df.TOTAL, 'b-')
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Descomponer la serie y graficar
advanced_decomposition = STL(df.TOTAL, period=4).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
ax1.plot(advanced_decomposition.observed, 'b-')
ax1.set_ylabel('Observado')
ax2.plot(advanced_decomposition.trend, 'b-')
ax2.set_ylabel('Tendencia')
ax3.plot(advanced_decomposition.seasonal, 'b-')
ax3.set_ylabel('Estacionalidad')
ax4.plot(advanced_decomposition.resid, 'b-')
ax4.set_ylabel('Residuos')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
fig.autofmt_xdate()
plt.xlabel('Periodo')
plt.tight_layout()
plt.show()

############################
# Prueba Dickey-Fuller (ADF) test (DA COMO RDO, SERIE ESTACIONARIA)
print('### Prueba Dickey-Fuller (ADF) test ###')
ADF_result = adfuller(df['TOTAL'])

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# # NO HACE FALTA YA QUE DIO ESTACIONARIO
# # Aplico 1er Diferenciacion (PARA PROBAR, AUNQUE YA DIO ESTACIONARIA)
# diff_primera = np.diff(df['TOTAL'], n=1)
# #
# # ADF_result = adfuller(diff_primera)
# #
# # print(f'ADF Statistic: {ADF_result[0]}')
# # print(f'p-value: {ADF_result[1]}')
#
print('### Autocorrelacion ###')
plot_acf(df['TOTAL'], lags=20, color='blue')  # con 30 aparecen algunos en la zona de confi.
plt.tight_layout()
plt.show()
#
# ##### Tampoco tiene sentido. deja de haber autocorrelacion.
# # plot_acf(diff_primera, lags=20)
# #
# # plt.tight_layout()
# # plt.show()
#
# # plt.plot(diff_primera)
# # plt.title('Differenced')
# # plt.xlabel('Timesteps')
# # plt.ylabel('Value')
# # plt.tight_layout()
# # plt.show()
#
print('### Autocorrelacion Parcial ###')
plot_pacf(df['TOTAL'], lags=20, color='blue')  # con 30 aparecen algunos en la zona de confi.
plt.tight_layout()
plt.show()
