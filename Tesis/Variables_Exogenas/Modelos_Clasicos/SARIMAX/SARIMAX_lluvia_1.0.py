from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

filename = 'molinetes_concat_linea_A_ordenados_por_fecha.csv'
df = pd.read_csv(filename)
print(df)


# Separar datos de entrenamiento y prueba
target = df['TOTAL']
exog = df[['PRECIPITACION']]

target_train = target[0:1366]
exog_train = exog[0:1366]

target_test = target[1366:1457]
exog_test = exog[0:1457]

# NAIVE seasonal forecast. Toma el ultimo seasonal ciclo y lo repite en al prediccion
naive_seasonal_forecast = target_train[-91:].values

# Crear y ajustar el modelo SARIMA
SARIMA_model = SARIMAX(target_train, exog=exog_train, order=(1, 0, 2), seasonal_order=(4, 0, 2, 7), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
print(SARIMA_model_fit.summary())


#Prediccion
# Predicción
SARIMA_pred = SARIMA_model_fit.get_prediction(start=1366, end=1456, exog=exog_test[1366:1457]).predicted_mean

# Crear DataFrame de prueba para almacenar los resultados de la predicción
test = pd.DataFrame({'actual': target_test})
test['SARIMA_pred'] = SARIMA_pred

print(test)


#Grafico
fig, ax = plt.subplots()

ax.plot(df.index, df['TOTAL'], 'b-', label='datos reales')
ax.plot(test.index, test['SARIMA_pred'], color='green', linestyle='--', label='SARIMA(1,0,2)(4,0,2,7)')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(test.index[0], test.index[-1], color='#808080', alpha=0.2)

ax.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()

#plt.show()

#############################
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
mape_error = mape(test.actual, test.SARIMA_pred)
smape_error = smape(test.actual, test.SARIMA_pred)
mae_error = mae(test.actual, test.SARIMA_pred)
mase_error = mase(test.actual, test.SARIMA_pred, naive_seasonal_forecast)

print('MAE: ', mae_error)
print('MAPE: ', mape_error)
print('MASE: ', mase_error)
print('SMAPE: ', smape_error)