# Importar libreriras
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

random.seed(42)

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

csv_file_name = "Con_Precipitacion_Feriado_Laborable.csv"

# Leer CSV e imprimir las primeras 5 líneas
df = pd.read_csv(csv_file_name)
print(df.head())

# Feature engineering
# conviertir la columna 'FECHA' a formato de fecha y hora. Extraer año, mes y día.
df['FECHA'] = pd.to_datetime(df['FECHA'])
df['Year'] = df['FECHA'].apply(lambda x: x.year)
df['Month'] = df['FECHA'].apply(lambda x: x.month)
df['Day'] = df['FECHA'].apply(lambda x: x.day)
# Generar características de estacionalidad semanal
df['DiaSemana'] = df['FECHA'].dt.dayofweek  # 0: Lunes, 1: Martes, ..., 6: Domingo
df['DiaAnio'] = df['FECHA'].dt.dayofyear
df['PROMEDIO_MOVIL'] = df['TOTAL'].rolling(window=7, min_periods=1).mean()  # Promedio móvil de 7 días

# Create X and y object
df = df.dropna()
# Separo datos en train and test
train = df[0:1366]
test = df[1366:1457]

# Separate the features and target variables for train and test sets
X_train = train[['Year', 'Month', 'Day', 'DiaSemana', 'DiaAnio', 'PROMEDIO_MOVIL','PRECIPITACION', 'FERIADO', 'LABORABLE']]
y_train = train['TOTAL']

X_test = test[['Year', 'Month', 'Day', 'DiaSemana', 'DiaAnio', 'PROMEDIO_MOVIL','PRECIPITACION', 'FERIADO', 'LABORABLE']]
y_test = test['TOTAL']
########################################

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'reg_lambda': [1, 0.01, 0.1]
}
# param_grid = {
#     'n_estimators': [2],
#     'learning_rate': [10e-4, 1.0],
#     'max_depth': [5],
# }

# Predicción y evaluación del modelo
xgb = XGBRegressor()
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

xgb_fcst = best_xgb.predict(X_test)
r2_score(list(y_test), list(xgb_fcst))

# Definicion funciones errores
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
mape_error = mape(y_test.values, xgb_fcst)
smape_error = smape(y_test.values, xgb_fcst)
mae_error = mae(y_test.values, xgb_fcst)
mae_n_error = mae_n(y_test.values, xgb_fcst)

print('MAE: ', mae_error)
print('MAPE: ', mape_error)
print('MAE_n ', mae_n_error)
print('SMAPE: ', smape_error)

# GRAFICOS
fig, ax = plt.subplots()

ax.plot(train.index, train['TOTAL'], 'b-', label='Datos reales')
ax.plot(test.index, test['TOTAL'], 'b-')
ax.plot(test.index, xgb_fcst, 'g-.', label='XGBoost')

ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(1366, 1457, color='#808080', alpha=0.2)

ax.legend(loc=2)

plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlim(0, 1460)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()
