import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import ExponentialSmoothing

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from itertools import product

import random

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

random.seed(0)


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

########################################
##SARIMA - LABORABLE
filename = 'Con_Laborable.csv'
df = pd.read_csv(filename)
print(df)

# Separar datos de entrenamiento y prueba
target = df['TOTAL']
exog = df[['LABORABLE']]

target_train = target[0:1366]
exog_train = exog[0:1366]

target_test = target[1366:1457]
exog_test = exog[0:1457]

# NAIVE seasonal forecast. Toma el ultimo seasonal ciclo y lo repite en al prediccion
naive_seasonal_forecast = target_train[-91:].values

# Crear y ajustar el modelo SARIMA
SARIMA_model = SARIMAX(target_train, exog=exog_train, order=(1, 0, 2), seasonal_order=(3, 0, 2, 7), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
print(SARIMA_model_fit.summary())

# Predicción
SARIMA_pred = SARIMA_model_fit.get_prediction(start=1366, end=1456, exog=exog_test[1366:1457]).predicted_mean

# Obtén las fechas para las predicciones (asumiendo que tienes una columna 'FECHA' en tu DataFrame original)
fechas_prediccion = df.loc[1366:1456, 'FECHA']

# Crea un nuevo DataFrame con las fechas y los valores predichos
predicciones_df = pd.DataFrame({'FECHA': fechas_prediccion, 'Prediccion': SARIMA_pred})

###############################################
#Holt Winter
filename = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filename)

#Se define el DataFrame en el conjunto de datos train y test
train = df['TOTAL'][0:1366]
test = df['TOTAL'][1366:1457]

# Aplicamos Holt-Winter #TODO ES CORRECTO LO QUE SE ESTA USANDO PARA CALCULAR???
mod_HW = ExponentialSmoothing(train, seasonal_periods=7,trend=None, seasonal='add').fit(optimized=1)
df['Holt_Winter'] = mod_HW.fittedvalues
print(df)

# Ajusto el tipo de dato en el dataset
df['FECHA'] = pd.to_datetime(df['FECHA'])
df['TOTAL'] = df['TOTAL'].astype(int)

# Predicción método de Holt
pred_HW= mod_HW.forecast(len(test))


predicciones_df = pd.DataFrame({'FECHA': fechas_prediccion, 'TOTAL': test ,'Pred_SARIMA': SARIMA_pred, 'Pred_HW': pred_HW })
print(predicciones_df)


# ##################################################
##PROPHET - PRECIPITACION LAG 1

filname = 'Con_Precipitacion_LAG1.csv'

#Leer CSV e imprimo las primeras 5 lineas
df1 = pd.read_csv(filname)
print(df1.head())

#Cambiamos nombres a las columnas
df1.columns = ['ds', 'y','PRECIPITACION_LAG1']
df1.head()

# Separar datos de entrenamiento y prueba
train = df1[0:1366]
test = df1[1366:1457]

#Ajuste de hiperparametros y validacion cruzada
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5, 0.9],
    'seasonality_prior_scale': [1.0, 10.0, 15.0, 20.0],
    'holidays_prior_scale': [1.0, 10.0, 15.0],
    'n_changepoints': [10],
    'weekly_seasonality': [True],
}

params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

maes = []

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
future['PRECIPITACION_LAG1'] = df1['PRECIPITACION_LAG1'].values[:len(future)]
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(91))

# Seleccionar las columnas 'ds' (fecha) y 'yhat' de 'forecast' y renombrar 'ds' a 'FECHA'
forecast_subset = forecast[['ds', 'yhat']]
forecast_subset.rename(columns={'ds': 'FECHA'}, inplace=True)

# Agregar la columna 'Pred_Prophet' (yhat de forecast_subset) a predicciones_df
predicciones_df['Pred_Prophet'] = forecast_subset['yhat']

print(predicciones_df)
##########################################
#XGBoost

csv_file_name = "linea_A_concat_xdia_total.csv"

# Listing 14-1. Importing the data
# Leer CSV e imprimir las primeras 5 líneas
df2 = pd.read_csv(csv_file_name)

# Seasonality variables
df2['FECHA'] = pd.to_datetime(df2['FECHA'])
df2['Year'] = df2['FECHA'].apply(lambda x: x.year)
df2['Month'] = df2['FECHA'].apply(lambda x: x.month)
df2['Day'] = df2['FECHA'].apply(lambda x: x.day)
# Generar características de estacionalidad semanal
df2['DiaSemana'] = df2['FECHA'].dt.dayofweek  # 0: Lunes, 1: Martes, ..., 6: Domingo
df2['DiaAnio'] = df2['FECHA'].dt.dayofyear
df2['PROMEDIO_MOVIL'] = df2['TOTAL'].rolling(window=7, min_periods=1).mean()  # Promedio móvil de 7 días

# Create X and y object
df2 = df2.dropna()
# Separo datos en train and test
train = df2[0:1366]
test = df2[1366:1457]

# Separate the features and target variables for train and test sets
X_train = train[['Year', 'Month', 'Day', 'DiaSemana', 'DiaAnio', 'PROMEDIO_MOVIL']]
y_train = train['TOTAL']

X_test = test[['Year', 'Month', 'Day', 'DiaSemana', 'DiaAnio', 'PROMEDIO_MOVIL']]
y_test = test['TOTAL']
########################################
param_grid = {
    'n_estimators': [10, 100],
    'learning_rate': [10e-4, 1.0],
    'max_depth': [0, 35],
}

xgb = XGBRegressor()
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

xgb_fcst = best_xgb.predict(X_test)
print(xgb_fcst)

# Crear un nuevo DataFrame con índices desde 1366 hasta 1456
nuevo_df = pd.DataFrame(index=range(1366, 1457))

# Agregar los resultados de xgb_fcst a una columna llamada 'Predicciones'
nuevo_df['Pred_XGBoost'] = xgb_fcst

# Mostrar el nuevo DataFrame
print(nuevo_df)

# Agregar la columna 'Pred_Prophet' (yhat de forecast_subset) a predicciones_df
predicciones_df['Pred_XGBoost'] = nuevo_df['Pred_XGBoost']

print(predicciones_df)

# Definir pesos para cada modelo (en este ejemplo, pesos iguales)
weight_sarima = 0.25
weight_holt_winter = 0.25
weight_prophet = 0.25
weight_xgboost = 0.25

# Calcular el promedio ponderado de las predicciones
weighted_ensemble_predictions = (
    weight_sarima * SARIMA_pred +
    weight_holt_winter * pred_HW +
    weight_prophet * forecast_subset['yhat'] +
    weight_xgboost * nuevo_df['Pred_XGBoost']
) / (weight_sarima + weight_holt_winter + weight_prophet + weight_xgboost)

# Agregar las predicciones ponderadas al DataFrame de prueba
predicciones_df['Pred_Ponderada'] = weighted_ensemble_predictions
print(predicciones_df)

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
mape_error = mape(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada)
smape_error = smape(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada)
mae_error = mae(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada)
mae_n_error = mae_n(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada)

print('MAE: ', mae_error)
print('MAPE: ', mape_error)
print('MAE_n: ', mae_n_error)
print('SMAPE: ', smape_error)



#GRAFICO
fig, ax = plt.subplots()

ax.plot(df.index, df['TOTAL'], 'b-', label='datos reales')
ax.plot(predicciones_df.index, predicciones_df['Pred_Ponderada'], color='green', linestyle='--', label='Pred_Ponderada')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(test.index[0], test.index[-1], color='#808080', alpha=0.2)

ax.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()