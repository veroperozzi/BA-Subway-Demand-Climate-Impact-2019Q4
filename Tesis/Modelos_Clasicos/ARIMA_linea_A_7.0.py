# Importar librerias
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

# Leer archivo CSV
filename = "linea_A_concat_xdia_total.csv"
df = pd.read_csv(filename)
print(df)

# Datos de entrenamiento y prueba
train = df['TOTAL'][0:1366]
test = df['TOTAL'][1366:1457]
######################################################################
# def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
#     results = []
#
#     for order in tqdm(order_list):
#         try:
#             model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False)
#         except:
#             continue
#
#         aic = model.aic
#         results.append([order, aic])
#
#     result_df = pd.DataFrame(results)
#     result_df.columns = ['(p,q)', 'AIC']
#
#     # Sort in ascending order, lower AIC is better
#     result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
#
#     return result_df
#
# ps = range(0, 4, 1)
# qs = range(0, 4, 1)
# d = 0
#
# order_list = list(product(ps, qs))
#
# train = df['TOTAL']
#
# result_df = optimize_ARIMA(train, order_list, d)
# print(result_df)
####################################################################


# Entrenar modelo ARIMA. orden p = 7 d = 0 y q = 7
model = SARIMAX(train, order=(7, 0, 7), simple_differencing=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())

# Grafico Diagnostico del Ruido
custom_styles = {'distplot_kwds': {'color': 'red'}, 'qqplot_kwds': {'line': 'r'}}
model_fit.plot_diagnostics(figsize=(10, 8))
plt.show()

# Test Ljung-Box
print('### Test Ljung-Box ###')
residuals = model_fit.resid
pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(pvalue)

# Realizar predicciones ARIMA
start_idx = test.index[0]
end_idx = test.index[-1]
ARIMA_pred = model_fit.get_prediction(start_idx, end_idx).predicted_mean.astype(float)
test = pd.DataFrame({'TOTAL': test.values, 'Pred_ARIMA': ARIMA_pred})
test['Fecha'] = pd.date_range(start='2019-10-01', periods=len(test))

# Imprimir comparación de datos reales y predicciones ARIMA
print('Comparativo Test y Prediccion ARIMA')
print(test)

# Grafico prediccion
fig, ax = plt.subplots()
ax.plot(df['FECHA'], df['TOTAL'], 'b-', label='datos reales')
ax.plot(test['Pred_ARIMA'], 'orange', linestyle='--', label='ARIMA(7,0,7)')
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(1366, 1458, color='#808080', alpha=0.2)
ax.legend(loc=2)
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlim(0, 1460)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# CALCULAMOS ERRORES
def mape(test_data, prediction):
    error = np.abs((test_data - prediction) / test_data)
    mape_value = np.mean(error) * 100
    normalized_mape = mape_value / 100
    return normalized_mape


def smape(test_data, prediction):
    absolute_error = np.abs(test_data - prediction)
    sum_absolute = np.abs(test_data) + np.abs(prediction)
    smape_value = np.mean(absolute_error / sum_absolute) * 100

    # Normaliz SMAPE to the range [0, 1]
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

# Calculo errores de predicción para el conjunto de prueba
mape_error = mape(test["TOTAL"], test['Pred_ARIMA'])
smape_error = smape(test["TOTAL"], test['Pred_ARIMA'])
mae_n_error = mae_n(test["TOTAL"], test['Pred_ARIMA'])
mae_error = mae(test["TOTAL"], test['Pred_ARIMA'])

# Imprimir errores
print('MAPE: ', mape_error)
print('SMAPE: ', smape_error)
print('MAE_n: ', mae_n_error)
print('MAE: ', mae_error)


