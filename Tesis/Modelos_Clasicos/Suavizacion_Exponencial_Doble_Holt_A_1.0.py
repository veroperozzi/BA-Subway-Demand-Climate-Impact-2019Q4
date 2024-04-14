#Importar librerias
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import Holt

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

#Leer CSV
filename = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filename)
print(df.head())

# Conjunto de datos train y test
train = df['TOTAL'][0:1366]
test = df['TOTAL'][1366:1457]

# Aplicar Suavización Exponencial de Holt
#model_holt = Holt(train).fit(smoothing_level=0.32, smoothing_slope=0.22, optimized=False)
model_holt = Holt(train).fit(optimized=True)
df['Holt'] = model_holt.fittedvalues
print(df)

# Cambio tipo de dato
df['FECHA'] = pd.to_datetime(df['FECHA'])
df['TOTAL'] = df['TOTAL'].astype(int)

# Realizar predicciones para Suavización Exponencial de Holt
pred_Holt = model_holt.forecast(len(test))

# Agregar predicciones al DataFrame
test = pd.DataFrame({'TOTAL': test.values, 'Pred_Holt': pred_Holt})
test['Fecha'] = pd.date_range(start='2019-10-01', periods=len(test))
print(test) # Imprimimos el DataFrame con las predicciones

# Definir errores
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

# Calculo errores
mape_error = mape(test['TOTAL'], test['Pred_Holt'])
smape_error = smape(test['TOTAL'], test['Pred_Holt'])
mae_n_error = mae_n(test['TOTAL'], test['Pred_Holt'])
mae_error = mae(test['TOTAL'], test['Pred_Holt'])

print('MAPE: ', mape_error)
print('SMAPE: ', smape_error)
print('MAE_n: ', mae_n_error)
print('MAE: ', mae_error)

#Vemos que valores uso el modelo ya que le pedimos que nos de los optimos
print(model_holt.summary())

#Puedo pedirle solo los parametros
print(model_holt.params_formatted)

# Vizualizamos set de datos origingal + Pred_ES
plt.figure(figsize=(12, 6))
plt.plot(df['TOTAL'],'b-', label='datos reales')
plt.plot(test['Pred_Holt'], 'yellow',linestyle='--', label='Pred_Holt')
plt.xlabel('Periodo', fontsize=14)
plt.ylabel('Total pasajeros linea A', fontsize=14)
plt.title('Linea A TOTAL de pasajeros, Suavizacion Exponencial|Holt', fontsize=18)
plt.xticks(np.arange(0, 1457, 365), [2016, 2017, 2018, 2019])
plt.axvspan(test.index[0], test.index[-1], color='#808080', alpha=0.2)
plt.legend(loc='best')
plt.show()

