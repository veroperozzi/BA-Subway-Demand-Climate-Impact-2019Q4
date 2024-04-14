import numpy as np
import pandas as pd
import plotly
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from itertools import product

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

filname = 'Con_Precipitacion_LAG1.csv'

#Leer CSV e imprimo las primeras 5 lineas
df = pd.read_csv(filname)
print(df.head())

#Cambiamos nombres a las columnas
df.columns = ['ds', 'y','PRECIPITACION_LAG1']
df.head()

# Separar datos de entrenamiento y prueba
train = df[0:1366]
test = df[1366:1457]

#Ajuste de hiperparametros y validacion cruzada
param_grid = {
    'changepoint_prior_scale': [0.019],
    'seasonality_prior_scale': [3.0],
    'holidays_prior_scale': [7.0],
    'n_changepoints':[8],
    'weekly_seasonality': [True],
}

params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

maes = []#TODO

for param in params:
    m = Prophet(**param)
    m.add_regressor('PRECIPITACION_LAG1')
    m.add_country_holidays(country_name='AR')
    m.fit(train)

    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='91 days', parallel='processes')
    df_p = performance_metrics(df_cv, rolling_window=1)
    maes.append(df_p['mae'].values[0])

tuning_results = pd.DataFrame(params)
tuning_results['mae'] = maes

best_params = params[np.argmin(maes)]
print(best_params)

m = Prophet(**best_params)
m.add_regressor('PRECIPITACION_LAG1')
m.add_country_holidays(country_name='AR')
m.fit(train)

#Prediccion
future = m.make_future_dataframe(periods=len(test), freq='D')
future['PRECIPITACION_LAG1'] = df['PRECIPITACION_LAG1'].values[:len(future)]
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(91))

test[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
print(test.head())

#last season naive forecast
test['baseline'] = train['y'][-91:].values
print(test.head())

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
mape_error = mape(test['y'], test['yhat'])
smape_error = smape(test['y'], test['yhat'])
mae_error = mae(test['y'], test['yhat'])
mase_error = mase(test['y'], test['yhat'],test['baseline'])

print('MAE: ', mae_error)
print('MAPE: ', mape_error)
print('MASE: ', mase_error)
print('SMAPE: ', smape_error)
