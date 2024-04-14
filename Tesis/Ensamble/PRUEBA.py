import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tools.sm_exceptions import ConvergenceWarning

import warnings

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import random

random.seed(0)


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)


tf.random.set_seed(42)
np.random.seed(42)

# index_col=0 Cual columna del CSV se utiliza como indice del DataFrame resultante.
train_df = pd.read_csv(
    '/home/vero/Documentos/proyectos_python/thesis_subway_climatic_impact1/Variables_Exogenas/Deep_Learning/LSTM/train.csv',
    index_col=0)
test_df = pd.read_csv(
    '/home/vero/Documentos/proyectos_python/thesis_subway_climatic_impact1/Variables_Exogenas/Deep_Learning/LSTM/test.csv',
    index_col=0)

print(train_df.shape, test_df.shape)


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

        exogenous_indices = [self.column_indices['PRECIPITACION'], self.column_indices['FERIADO'], self.column_indices['LABORABLE']]

        # Selección de las columnas de las variables exógenas
        exogenous_inputs = tf.gather(inputs, exogenous_indices, axis=-1)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        exogenous_inputs.set_shape([None, self.input_width, len(exogenous_indices)])

        return (inputs, exogenous_inputs), labels

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


# Definir las columnas que serán utilizadas como entrada y salida
input_columns = ['TOTAL', 'PRECIPITACION', 'FERIADO', 'LABORABLE', 'day_sin', 'day_cos']
output_columns = ['TOTAL']

multi_window = DataWindow(input_width=90, label_width=90, shift=1, label_columns=output_columns)

lstm_model = Sequential([
    Concatenate(axis=-1), # Combinar secuencia de serie de tiempo y secuencia de variables exógenas
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
print(predictions)
# Invertir el escalado de las predicciones
predictions_original_scale = scaler.inverse_transform(predictions)

# predictions_original_scale ahora contiene las predicciones en la escala original
# Puedes imprimir o utilizar predictions_original_scale según sea necesario
print(predictions_original_scale)

# # Definir pesos para cada modelo (en este ejemplo, pesos iguales)
# weight_sarima = 0.25
# weight_holt_winter = 0.25
# weight_prophet = 0.25
# weight_xgboost = 0.25
#
# # Calcular el promedio ponderado de las predicciones
# weighted_ensemble_predictions = (
#     weight_sarima * SARIMA_pred +
#     weight_holt_winter * pred_HW +
#     weight_prophet * forecast_subset['yhat'] +
#     weight_xgboost * nuevo_df['Pred_XGBoost']
# ) / (weight_sarima + weight_holt_winter + weight_prophet + weight_xgboost)
#
# # Agregar las predicciones ponderadas al DataFrame de prueba
# predicciones_df['Pred_Ponderada'] = weighted_ensemble_predictions
# print(predicciones_df)
#
# #############################
#
# #CALCULAMOS ERRORES
# def mape(test_data, prediction):
#     error = np.abs((test_data - prediction) / test_data)
#     mape_value = np.mean(error) * 100
#     normalized_mape = mape_value / 100
#     return normalized_mape
#
#
# def smape(test_data, prediction):
#     absolute_error = np.abs(test_data - prediction)
#     sum_absolute = np.abs(test_data) + np.abs(prediction)
#     smape_value = np.mean(absolute_error / sum_absolute) * 100
#
#     # Normalize SMAPE to the range [0, 1]
#     normalized_smape = smape_value / 200  # Since the maximum SMAPE is 200%
#
#     return normalized_smape
#
# def mae(test_data, prediction):
#     absolute_error = np.abs(test_data - prediction)
#     mean_absolute_error = np.mean(absolute_error)
#
#     # Normalize MAE
#     max_value = np.max(test_data)
#     normalized_mae = mean_absolute_error / max_value
#
#     return normalized_mae
#
# def mase(test_data, prediction, naive_forecast):
#     error = np.abs(test_data - prediction)
#     mean_absolute_error = np.mean(error)
#
#     naive_error = np.abs(test_data - naive_forecast)
#     mean_naive_error = np.mean(naive_error)
#
#     mase_value = mean_absolute_error / mean_naive_error
#
#     # Normalize MASE to the range [0, 1]
#     normalized_mase = mase_value / (1 + mase_value)
#
#     return normalized_mase
#
# # Calcular errores de predicción para el conjunto de prueba
# mape_error = mape(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada)
# smape_error = smape(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada)
# mae_error = mae(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada)
# mase_error = mase(predicciones_df.TOTAL, predicciones_df.Pred_Ponderada, naive_seasonal_forecast)
#
# print('MAE: ', mae_error)
# print('MAPE: ', mape_error)
# print('MASE: ', mase_error)
# print('SMAPE: ', smape_error)
#
#
#
# #GRAFICO
# fig, ax = plt.subplots()
#
# ax.plot(df.index, df['TOTAL'], 'b-', label='datos reales')
# ax.plot(predicciones_df.index, predicciones_df['Pred_Ponderada'], color='green', linestyle='--', label='Pred_Ponderada')
# plt.xticks(np.arange(0, 1460, 365), [2016, 2017, 2018, 2019])
# ax.set_xlabel('Periodo')
# ax.set_ylabel('Total Pasajeros Linea A')
# ax.axvspan(test.index[0], test.index[-1], color='#808080', alpha=0.2)
#
# ax.legend(loc=2)
#
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show()