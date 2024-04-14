import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

csv_file_name = "Con_lluvia_y_feriados.csv"

#Listing 14-1. Importing the data
# Leer CSV e imprimir las primeras 5 líneas
df = pd.read_csv(csv_file_name)
print(df.head())

#Listing 14-2. Feature engineering
# Seasonality variables
df['FECHA'] = pd.to_datetime(df['FECHA'])
df['Year'] = df['FECHA'].apply(lambda x: x.year)
df['Month'] = df['FECHA'].apply(lambda x: x.month)
df['Day'] = df['FECHA'].apply(lambda x: x.day)

# Adding a week of lagged data
df['L1'] = df['TOTAL'].shift(1)
df['L2'] = df['TOTAL'].shift(2)
df['L3'] = df['TOTAL'].shift(3)
df['L4'] = df['TOTAL'].shift(4)
df['L5'] = df['TOTAL'].shift(5)
df['L6'] = df['TOTAL'].shift(6)
df['L7'] = df['TOTAL'].shift(7)

#Listing 14-3. Fitting the default Random Forest Regressor
# Create X and y object
df = df.dropna()
# Separo datos en train and test
train = df[0:1366]
test = df[1366:1457]

# Separate the features and target variables for train and test sets
X_train = train[['Year', 'Month', 'Day', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']]
y_train = train['TOTAL']

X_test = test[['Year', 'Month', 'Day', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']]
y_test = test['TOTAL']

#Entreno modelo y hago prediccion
my_rf = RandomForestRegressor()
my_rf.fit(X_train, y_train)
fcst = my_rf.predict(X_test)

#Evaluamos prediccion
from sklearn.metrics import r2_score
r2 = r2_score(list(y_test), list(fcst))
print(r2)


#last season naive forecast
test['baseline'] = train['TOTAL'][-85:].values
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
mape_error = mape(y_test.values, fcst)
smape_error = smape(y_test.values, fcst)
mae_error = mae(y_test.values, fcst)
mase_error = mase(y_test.values, fcst,test['baseline'])

print('MAE: ', mae_error)
print('MAPE: ', mape_error)
print('MASE: ', mase_error)
print('SMAPE: ', smape_error)

#GRAFICOS
fig, ax = plt.subplots()

ax.plot(train.index, train['TOTAL'],'b-',label='Datos reales')
ax.plot(test.index, test['TOTAL'],'b-')
ax.plot(test.index, fcst, 'g-.', label='Random Forest')

ax.set_xlabel('Periodo')
ax.set_ylabel('Total Pasajeros Linea A')
ax.axvspan(1366,1457, color='#808080', alpha=0.2)

ax.legend(loc=2)

plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
ax.set_xlim(0, 1460)
#plt.fill_between(x=test.index, y1=test['yhat_lower'], y2=test['yhat_upper'], color='lightblue')

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

