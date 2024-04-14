# Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import product

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

csv_file_name = "linea_A_concat_xdia_total.csv"

# Leer CSV e imprimo las primeras 5 lineas
df = pd.read_csv(csv_file_name)
print(df.head())

# Imprimo ultimas 5 lineas
print(df.tail())

# Cambiamos nombres a las columnas
df.columns = ['ds', 'y']
df.head()

# Separar datos de entrenamiento y prueba
train = df[0:1366]
test = df[1366:1457]

# Definición de la cuadrícula de hiperparámetros
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5, 0.9],
    'seasonality_prior_scale': [1.0, 10.0, 15.0, 20.0],
    'holidays_prior_scale': [1.0, 10.0, 15.0],
    'n_changepoints': [10],
    'weekly_seasonality': [True],
}

# Generar todas las combinaciones posibles de hiperparámetros
params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

# Almacenará los MAEs de cada conjunto de parámetros
maes = []

# Iterar sobre todas las combinaciones de hiperparámetros
for param in params:
    m = Prophet(**param)
    m.add_country_holidays(country_name='AR')
    m.fit(train)

    # Realizar validación cruzada y calcular el MAE
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='91 days', parallel='processes')
    df_p = performance_metrics(df_cv, rolling_window=1)
    maes.append(df_p['mae'].values[0])

tuning_results = pd.DataFrame(params)
tuning_results['mae'] = maes

# Encontrar los mejores parámetros
best_params = params[np.argmin(maes)]
print(best_params)

m = Prophet(**best_params)
m.add_country_holidays(country_name='AR')
m.fit(train)

# Prediccion
future = m.make_future_dataframe(periods=len(test), freq='D')
forecast = m.predict(future)
# Imprimir las últimas predicciones
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(91))

# Asignar las predicciones al conjunto de prueba
test[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
print(test.head())


# Definir funcion de errores
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
mape_error = mape(test['y'], test['yhat'])
smape_error = smape(test['y'], test['yhat'])
mae_n_error = mae_n(test['y'], test['yhat'])
mae_error = mae(test['y'], test['yhat'])

print('MAPE: ', mape_error)
print('SMAPE: ', smape_error)
print('MAE_n: ', mae_n_error)
print('MAE: ', mae_error)

# GRAFICO
fig, ax = plt.subplots()
ax.plot(train['y'], 'b-', label='Datos reales')
ax.plot(test['y'], 'b-')
ax.plot(test['yhat'], 'g-.', label='Prophet')
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(1366, 1457, color='#808080', alpha=0.2)
ax.legend(loc=2)
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlim(0, 1460)
# plt.fill_between(x=test.index, y1=test['yhat_lower'], y2=test['yhat_upper'], color='lightblue')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

prophet_components_fig = m.plot_components(forecast)
plt.show()
