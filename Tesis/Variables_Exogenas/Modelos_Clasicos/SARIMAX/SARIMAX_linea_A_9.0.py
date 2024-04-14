import warnings
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

warnings.filterwarnings('ignore')

filename = 'molinetes_concat_linea_A_ordenados_por_fecha.csv'
df = pd.read_csv(filename)
print(df)

# # Analizamos Variable TOTAL pasajeros y 3 Variables Exogenas
# fig, axes = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=(11, 6))
#
# # Nombres de las columnas que se van a graficar
# column_names = ["TOTAL", "PRECIPITACION", "FERIADO"]

# # Iterar sobre los subplots y graficar las columnas deseadas
# for i, ax in enumerate(axes.flatten()):
#     data = df[column_names[i]]
#
#     ax.plot(data, color='blue', linewidth=0.3)
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.spines['top'].set_alpha(0)
#     ax.tick_params(labelsize=6)
#     # Agregar el título a la izquierda de cada gráfico
#     ax.text(-0.1, 0.5, column_names[i], transform=ax.transAxes,
#             rotation=90, va='center', ha='left', fontsize=4)
#
# plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
# plt.subplots_adjust(top=0.85, hspace=0.1)
# fig.autofmt_xdate()
# plt.tight_layout()
# #plt.show()
#
# # Dickey-Fuller (ADF) test
target = df['TOTAL']
exog = df[['PRECIPITACION', 'FERIADO']]

#
# ad_fuller_result = adfuller(target)
# print(f'ADF Statistic: {ad_fuller_result[0]}')
# print(f'p-value: {ad_fuller_result[1]}')


# Como se rechaza la H0 con lo cual la serie es estacionaria. d=0 and D = 0

# Definimos el SARIMAX optimo
def optimize_SARIMAX(endog: Union[pd.Series, list], exog: Union[pd.Series, list], order_list: list, d: int, D: int,
                     s: int) -> pd.DataFrame:
    results = []

    for order in tqdm(order_list):
        try:
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue

        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df


p = range(0,8, 1)
d = 0
q = range(0,1, 1)
P = range(0,1, 1)
D = 0
Q = range(0,1, 1)
s = 7

parameters = product(p, q, P, Q)
parameters_list = list(parameters)

target_train = target[0:1366]
exog_train = exog[0:1366]

print(type(target_train))
print(type(exog_train))

result_df = optimize_SARIMAX(target_train, exog_train, parameters_list, d, D, s)
print(result_df)

#Entrena modelo
best_model = SARIMAX(target_train, exog_train, order=(3,0,3), seasonal_order=(2,0,3,7), simple_differencing=False)
best_model_fit = best_model.fit(disp=False)

print(best_model_fit.summary())

#Analizamos los residuos del Modelo
best_model_fit.plot_diagnostics(figsize=(10,8))
plt.show()

#Ljung-Box
residuals = best_model_fit.resid
pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(pvalue)

#Forcasting el proximo pasos multiples veces
def recursive_forecast(endog: Union[pd.Series, list], exog: Union[pd.Series, list], train_len: int, horizon: int,
                       window: int, method: str) -> list:
    total_len = train_len + horizon

    if method == 'last':
        pred_last_value = []

        for i in range(train_len, total_len, window):
            last_value = endog[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(window))

        return pred_last_value

    elif method == 'SARIMAX':
        pred_SARIMAX = []

        for i in range(train_len, total_len, window):
            model = SARIMAX(endog[:i], exog[:i], order=(3, 0, 3), seasonal_order=(2, 0, 3, 7),
                            simple_differencing=False)
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMAX.extend(oos_pred)

        return pred_SARIMAX


#predict the next timestep over a certain period of time.
target_train = target[0:1366]
target_test = target[1366:1457]

pred_df = pd.DataFrame({'actual': target_test})

TRAIN_LEN = len(target_train)
HORIZON = len(target_test)
WINDOW = 1

pred_last_value = recursive_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_SARIMAX = recursive_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, 'SARIMAX')

pred_df['pred_last_value'] = pred_last_value
pred_df['pred_SARIMAX'] = pred_SARIMAX

print(pred_df)


#Grafico
fig, ax = plt.subplots()

ax.plot(df.index, df['TOTAL'], 'b-', label='datos reales')
ax.plot(pred_df.index, pred_df['pred_SARIMAX'], color='green', linestyle='--', label='SARIMA(3,0,3)(2,0,3,7)')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(pred_df.index[0], pred_df.index[-1], color='#808080', alpha=0.2)

ax.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()

plt.show()


# #############################
#CALCULAMOS ERRORES
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

def mae(test_data, prediction):
    absolute_error = np.abs(test_data - prediction)
    mean_absolute_error = np.mean(absolute_error)

    # Normalize MAE
    max_value = np.max(test_data)
    normalized_mae = mean_absolute_error / max_value

    return normalized_mae


def mase(test_data, prediction, naive_forecast):
    error = np.abs(test_data - prediction)
    mean_absolute_error = np.mean(error)

    naive_error = np.abs(test_data - naive_forecast)
    mean_naive_error = np.mean(naive_error)

    mase_value = mean_absolute_error / mean_naive_error

    # Normalize MASE to the range [0, 1]
    normalized_mase = mase_value / (1 + mase_value)

    return normalized_mase

# Calcular errores de predicción para el conjunto de prueba
mape_error = mape(pred_df.actual, pred_df.pred_SARIMAX)
smape_error = smape(pred_df.actual, pred_df.pred_SARIMAX)
mae_error = mae(pred_df.actual, pred_df.pred_SARIMAX)
mase_error = mase(pred_df.actual, pred_df.pred_SARIMAX, pred_df.pred_last_value)

print('MAPE: ', mape_error)
print('SMAPE: ', smape_error)
print('MAE: ', mae_error)
print('MASE: ', mase_error)
