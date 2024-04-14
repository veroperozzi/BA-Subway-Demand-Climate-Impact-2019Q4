import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import ExponentialSmoothing

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

########################################
##SARIMA
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

# Aplicamos Holt-Winter
mod_HW = ExponentialSmoothing(train, seasonal_periods=7,trend=None, seasonal='add').fit(optimized=1)
df['Holt_Winter'] = mod_HW.fittedvalues
print(df)

# Ajusto el tipo de dato en el dataset
df['FECHA'] = pd.to_datetime(df['FECHA'])
df['TOTAL'] = df['TOTAL'].astype(int)

# Predicción método de Holt
pred_HW= mod_HW.forecast(len(test))

##################################################

predicciones_df = pd.DataFrame({'FECHA': fechas_prediccion, 'TOTAL': test ,'Pred_SARIMA': SARIMA_pred, 'Pred_HW': pred_HW })
print(predicciones_df)

# Definir pesos para cada modelo (en este ejemplo, pesos iguales)
weight_sarima = 0.5
weight_holt_winter = 0.5

# Calcular el promedio ponderado de las predicciones
weighted_ensemble_predictions = (
    weight_sarima * SARIMA_pred +
    weight_holt_winter * pred_HW
) / (weight_sarima + weight_holt_winter)

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
ax.plot(predicciones_df.index, predicciones_df['Pred_Ponderada'], color='green', linestyle='--')
plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(test.index[0], test.index[-1], color='#808080', alpha=0.2)

ax.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()
