# Importar librerias
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from itertools import product
from typing import Union
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Leer CSV
filename = "linea_A_concat_xdia_total.csv"
df = pd.read_csv(filename)
df['FECHA'] = pd.to_datetime(df['FECHA'])
df.info()
# Test Estasionariedada: Si es estacionaria entonces d = 0

# Separar datos de entrenamiento y prueba
train = df.iloc[0:1366]
test = df.iloc[1366:1457]

test['SARIMA_pred'] = np.nan  # Prepara la columna para las predicciones

print(train)
print(test)

##########################################################
#Defino funicon de Optimizacion
# def optimize_SARIMA(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
#     results = []
#
#     for order in tqdm(order_list):
#         try:
#             model = SARIMAX(
#                 endog,
#                 order=(order[0], d, order[1]),
#                 seasonal_order=(order[2], D, order[3], s),
#                 simple_differencing=False).fit(disp=False)
#         except:
#             continue
#
#         aic = model.aic
#         results.append([order, aic])
#
#     result_df = pd.DataFrame(results)
#     result_df.columns = ['(p,q,P,Q)', 'AIC']
#
#     # Sort in ascending order, lower AIC is better
#     result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
#
#     return result_df
#
# # Optimización y Entrenamiento de Modelo SARIMA
# ps = range(0, 7, 1)
# qs = range(0, 7, 1)
# Ps = range(0, 7, 1)
# Qs = range(0, 7, 1)
# SARIMA_order_list = list(product(ps, qs, Ps, Qs))
# d = 0
# D = 0
# s = 7
#
# SARIMA_result_df = optimize_SARIMA(train, SARIMA_order_list, d, D, s)
# print(SARIMA_result_df)
####################################################################

# Especificacion del modelo SARIMA. orden p = 3 d = 0 y q = 3  y P = 2, D = 0 Q = 3 y m = 7
SARIMA_model = SARIMAX(train['TOTAL'], order=(3, 0, 3), seasonal_order=(2, 0, 3, 7), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
print(SARIMA_model_fit.summary())

# Diagnóstico y Evaluación
SARIMA_model_fit.plot_diagnostics(figsize=(10, 8))
plt.show()
residuals = SARIMA_model_fit.resid
pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(pvalue)

# Prediccion
start = test.index[0]
end = test.index[-1]
predictions = SARIMA_model_fit.predict(start=start, end=end, dynamic=False)

# Añadir predicciones al DataFrame de test
test['SARIMA_pred'] = predictions.values
# Mostrar los resultados
print(test[['TOTAL', 'SARIMA_pred']])

# Grafico
fig, ax = plt.subplots()
ax.plot(df.index, df['TOTAL'], 'b-', label='datos reales')
ax.plot(test.index, test['SARIMA_pred'], color='green', linestyle='--', label='SARIMA(3,0,3)(2,0,3,7)')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(test.index[0], test.index[-1], color='#808080', alpha=0.2)
ax.legend(loc=2)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

#############################
# Definicion de funcion de ERRORES
def mape(test_data, prediction):
    error = np.abs((test_data - prediction) / test_data)
    mape_value = np.mean(error) * 100
    normalized_mape = mape_value / 100
    return normalized_mape

def smape(test_data, prediction):
    absolute_error = np.abs(test_data - prediction)
    sum_absolute = np.abs(test_data) + np.abs(prediction)
    smape_value = np.mean(absolute_error / sum_absolute) * 100

    # Normalize SMAPE to the range [0, 1]
    normalized_smape = smape_value / 200  # Since the maximum SMAPE is 200%

    return normalized_smape

def mae_n(test_data, prediction):
    absolute_error = np.abs(test_data - prediction)
    mean_absolute_error = np.mean(absolute_error)

    # Normalize MAE
    max_value = np.max(test_data)
    normalized_mae = mean_absolute_error / max_value

    return normalized_mae

def mae(test_data, prediction):
    return np.mean(np.abs(test_data - prediction))

# Calcular errores de predicción para el conjunto de prueba
mape_error = mape(test['TOTAL'], test['SARIMA_pred'])
smape_error = smape(test['TOTAL'], test['SARIMA_pred'])
mae_n_error = mae_n(test['TOTAL'], test['SARIMA_pred'])
mae_error = mae(test['TOTAL'], test['SARIMA_pred'])

# Imprimir resultados
print('MAPE: ', mape_error)
print('SMAPE: ', smape_error)
print('MAE_n: ', mae_n_error)
print('MAE: ', mae_error)