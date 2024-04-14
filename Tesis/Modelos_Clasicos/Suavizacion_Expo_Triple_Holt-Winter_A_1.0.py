#Suavizacion Exponencia Triple con NIVEL, TENDENCIA Y ESTACIONALIDAD
#Importar librerias
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf  # grafico autocorrelacion
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import ExponentialSmoothing

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

#Leer CSV
filename = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filename)

#Definir el conjunto de datos train y test
train = df['TOTAL'][0:1366]
test = df['TOTAL'][1366:1457]

# Determinar Estacionalidad de la serie de tiempo|Metodo Holt|StatsModels.
plt.rc('figure', figsize=(10, 6))
plot_acf(df['TOTAL'].values, fft=1)
plt.title('Autocorrelacion', fontsize=17)
plt.xlabel('Periodos')
plt.ylabel('Pasajeros Linea A')
plt.show()

# Aplicamos Holt-Winter
mod_HW = ExponentialSmoothing(train, seasonal_periods=7,trend=None, seasonal='add').fit(optimized=1)
df['Holt_Winter'] = mod_HW.fittedvalues
print(df)

# Cambio tipo de dato
df['FECHA'] = pd.to_datetime(df['FECHA'])
df['TOTAL'] = df['TOTAL'].astype(int)

# Predicción método de Holt
pred_HW= mod_HW.forecast(len(test))

# Agregamos las predicciones al DataFrame
test = pd.DataFrame({'TOTAL': test, 'Pred_HW': pred_HW})
test['Fecha'] = pd.date_range(start='2019-10-01', periods=len(test))
print(test) # Imprimimos el DataFrame con las predicciones

# Defino funcion errores
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
mape_error = mape(test['TOTAL'], test['Pred_HW'])
smape_error = smape(test['TOTAL'], test['Pred_HW'])
mae_n_error = mae_n(test['TOTAL'], test['Pred_HW'])
mae_error = mae(test['TOTAL'], test['Pred_HW'])

print('MAPE: ', mape_error)
print('SMAPE: ', smape_error)
print('MAE_n: ', mae_error)
print('MAE: ', mae_error)

#Vemos que valores uso el modelo ya que le pedimos que nos de los optimos
print(mod_HW.summary())

#Puedo pedirle solo los parametros
print(mod_HW.params_formatted)


# Vizualizamos set de datos origingal + Pred_ES
plt.figure(figsize=(12, 6))
plt.plot(df['TOTAL'],'b-', label='datos reales')
plt.plot(test['Pred_HW'], 'green',linestyle='--', label='Pred_HW')
plt.xlabel('Periodo', fontsize=14)
plt.ylabel('Total pasajeros linea A', fontsize=14)
plt.title('Linea A TOTAL de pasajeros,Suavizacion Exponencial Triple|Holt Winter', fontsize=18)
plt.xticks(np.arange(0, 1457, 365), [2016, 2017, 2018, 2019])
plt.legend(loc='best')
plt.axvspan(test.index[0], test.index[-1], color='#808080', alpha=0.2)
plt.show()
