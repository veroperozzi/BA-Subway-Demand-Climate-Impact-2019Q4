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

import warnings
import datetime
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

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

############################################
#LSTM - 3 Variables

filename = 'linea_A_concat_xdia_total.csv'
df3 = pd.read_csv(filename)

# Convierte columna fecha en valores Datatime
df3['FECHA'] = pd.to_datetime(df3['FECHA'])

timestamp_s = pd.to_datetime(df3['FECHA']).map(datetime.datetime.timestamp)

fft = tf.signal.rfft(df3['TOTAL'])
f_per_dataset = np.arange(0, len(fft))

n_sample_h = len(df3['TOTAL'])
hours_per_week = 24 * 7
weeks_per_dataset = n_sample_h / hours_per_week

f_per_week = f_per_dataset / weeks_per_dataset

# Eliminar las columnas innecesarias
df3 = df3.drop(['FECHA'], axis=1)
# Dividimos el dataset en test y train
n = len(df3)

# Split train:test
train_df = df3[0:1366]
test_df = df3[1366:1457]

# Imprimir dimenciones
print(train_df.shape)
print(test_df.shape)

scaler = MinMaxScaler()
scaler.fit(train_df)

train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

train_df.describe().transpose()

print(train_df)
print(test_df)

# DEFINIMOS DATA WINDOWS
class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, test_df=test_df,
                 label_columns=None):

        self.train_df = train_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='TOTAL'):
        inputs, _ = self.sample_batch

        plt.figure(figsize=(12, 6))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f'{plot_col} [scaled]')

        # Plot the complete dataset in blue
        combined_df = pd.concat([self.train_df, self.test_df])
        plt.plot(combined_df.index, combined_df[plot_col], label='Datos reales', color='blue')

        if model is not None:
            predictions = model(inputs)
            test_start = len(self.train_df)  # Index where test data starts
            test_end = len(combined_df) - 1  # Index of the last data point
            test_index = combined_df.index[
                         test_end - self.label_width + 1:test_end + 1]  # Index of the last label_width days
            plt.plot(test_index, predictions[0, -self.label_width:, plot_col_index],
                     label='LSTM', color='green', linestyle='--')

        plt.legend()

        # Adding custom x-axis ticks and labels
        plt.xticks(np.arange(0, len(combined_df), 365), [2016, 2017, 2018, 2019])

        # Adding x-axis and y-axis labels
        plt.xlabel('Periodo')
        plt.ylabel('Total Pasajeros Linea A')

        # Adding shaded region for the test period
        plt.axvspan(self.test_df.index[0], self.test_df.index[-1], color='#808080', alpha=0.2)

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result


####################################
def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='loss', patience=patience, mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=max_epochs,
                        callbacks=[early_stopping])

    return history


multi_window = DataWindow(input_width=90, label_width=90, shift=1, label_columns=['TOTAL'])

lstm_model = Sequential([
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),
    Dense(units=1)
])

# Entrenamos Modelo
history = compile_and_fit(lstm_model, multi_window)

performance = {}
performance['LSTM'] = lstm_model.evaluate(multi_window.test, verbose=0)


# Realizar predicciones en el conjunto de prueba
predictions = lstm_model.predict(multi_window.test)

# Crear un DataFrame vacío
resultados_df = pd.DataFrame(index=range(1366, 1456))

# Asignar los valores de predictions a una columna llamada 'Predicciones'
resultados_df['Predicciones'] = predictions.flatten()  # Flatten para convertirlo en una sola columna

# Puedes imprimir o trabajar con resultados_df según sea necesario
print(resultados_df)

# Invertir el escalado de la columna 'Predicciones'
resultados_df['Predicciones'] = scaler.inverse_transform(resultados_df[['Predicciones']])

# Ahora, resultados_df contiene los resultados de la predicción en la escala original
print(resultados_df)

# Aquí seleccionamos el rango de predicciones que deseas agregar (índices 1366 a 1455)
predicciones_a_agregar = resultados_df.loc[1366:1455, ['Predicciones']]

# Luego, asignamos estas predicciones al DataFrame predicciones_df en una nueva columna 'Pred_LSTM'
predicciones_df['Pred_LSTM'] = predicciones_a_agregar['Predicciones']

# Ahora predicciones_df contiene las predicciones desde el índice 1366 hasta 1455 en la columna 'Pred_LSTM'
print(predicciones_df)

# Completar los valores NaN en predicciones_df con la propagación hacia adelante
predicciones_df.fillna(method='ffill', inplace=True)

# Ahora predicciones_df no tiene valores NaN y los NaN se han completado con los valores de la celda anterior
print(predicciones_df)

###############################################
#GRU
# DEFINIMOS DATA WINDOWS
class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, test_df=test_df,
                 label_columns=None):

        self.train_df = train_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='TOTAL'):
        inputs, _ = self.sample_batch

        plt.figure(figsize=(12, 6))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f'{plot_col} [scaled]')

        # Plot the complete dataset in blue
        combined_df = pd.concat([self.train_df, self.test_df])
        plt.plot(combined_df.index, combined_df[plot_col], label='Datos reales', color='blue')

        if model is not None:
            predictions = model(inputs)
            test_start = len(self.train_df)  # Index where test data starts
            test_end = len(combined_df) - 1  # Index of the last data point
            test_index = combined_df.index[
                         test_end - self.label_width + 1:test_end + 1]  # Index of the last label_width days
            plt.plot(test_index, predictions[0, -self.label_width:, plot_col_index],
                     label='GRU', color='green', linestyle='--')

        plt.legend()

        # Adding custom x-axis ticks and labels
        plt.xticks(np.arange(0, len(combined_df), 365), [2016, 2017, 2018, 2019])

        # Adding x-axis and y-axis labels
        plt.xlabel('Periodo')
        plt.ylabel('Total Pasajeros Linea A')

        # Adding shaded region for the test period
        plt.axvspan(self.test_df.index[0], self.test_df.index[-1], color='#808080', alpha=0.2)

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result


####################################
def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='loss', patience=patience, mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=max_epochs,
                        callbacks=[early_stopping])

    return history


multi_window = DataWindow(input_width=90, label_width=90, shift=1, label_columns=['TOTAL'])

GRU_model = Sequential([
    GRU(64, return_sequences=True),
    GRU(64, return_sequences=True),
    GRU(64, return_sequences=True),
    GRU(64, return_sequences=True),
    GRU(64, return_sequences=True),
    Dense(units=1)
])

# Entrenamos Modelo
history = compile_and_fit(GRU_model, multi_window)

performance = {}
performance['GRU'] = GRU_model.evaluate(multi_window.test, verbose=0)

# Realizar predicciones en el conjunto de prueba
predictions2 = GRU_model.predict(multi_window.test)

# Crear un DataFrame vacío
resultados_df2 = pd.DataFrame(index=range(1366, 1456))

# Asignar los valores de predictions a una columna llamada 'Predicciones'
resultados_df2['Predicciones2'] = predictions.flatten()  # Flatten para convertirlo en una sola columna

# Puedes imprimir o trabajar con resultados_df según sea necesario
print(resultados_df2)

# Invertir el escalado de la columna 'Predicciones'
resultados_df2['Predicciones2'] = scaler.inverse_transform(resultados_df2[['Predicciones2']])

# Ahora, resultados_df contiene los resultados de la predicción en la escala original
print(resultados_df2)

# Aquí seleccionamos el rango de predicciones que deseas agregar (índices 1366 a 1455)
predicciones_a_agregar = resultados_df2.loc[1366:1455, ['Predicciones2']]

# Luego, asignamos estas predicciones al DataFrame predicciones_df en una nueva columna 'Pred_LSTM'
predicciones_df['Pred_GRU'] = predicciones_a_agregar['Predicciones2']

# Ahora predicciones_df contiene las predicciones desde el índice 1366 hasta 1455 en la columna 'Pred_LSTM'
print(predicciones_df)

# Completar los valores NaN en predicciones_df con la propagación hacia adelante
predicciones_df.fillna(method='ffill', inplace=True)

# Ahora predicciones_df no tiene valores NaN y los NaN se han completado con los valores de la celda anterior
print(predicciones_df)


# Definir pesos para cada modelo (en este ejemplo, pesos iguales)
weight_sarima = 0.10
weight_holt_winter = 0.10
weight_prophet = 0.35
weight_xgboost = 0.37
weight_LSTM = 0.09
weight_GRU = 0.09

# Calcular el promedio ponderado de las predicciones
weighted_ensemble_predictions = (
    weight_sarima * SARIMA_pred +
    weight_holt_winter * pred_HW +
    weight_prophet * forecast_subset['yhat'] +
    weight_xgboost * nuevo_df['Pred_XGBoost'] +
    weight_LSTM * predicciones_df['Pred_LSTM'] +
    weight_GRU * predicciones_df['Pred_GRU']
) / (weight_sarima + weight_holt_winter + weight_prophet + weight_xgboost + weight_LSTM + weight_GRU)

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