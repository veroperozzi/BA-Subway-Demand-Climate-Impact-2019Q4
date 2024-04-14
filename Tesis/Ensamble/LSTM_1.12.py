import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import warnings

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

filename = 'linea_A_concat_xdia_total.csv'
df = pd.read_csv(filename)

# Convierte columna fecha en valores Datatime
df['FECHA'] = pd.to_datetime(df['FECHA'])

timestamp_s = pd.to_datetime(df['FECHA']).map(datetime.datetime.timestamp)

fft = tf.signal.rfft(df['TOTAL'])
f_per_dataset = np.arange(0, len(fft))

n_sample_h = len(df['TOTAL'])
hours_per_week = 24 * 7
weeks_per_dataset = n_sample_h / hours_per_week

f_per_week = f_per_dataset / weeks_per_dataset

# Eliminar las columnas innecesarias
df = df.drop(['FECHA'], axis=1)
# Dividimos el dataset en test y train
n = len(df)

# Split train:test
train_df = df[0:1366]
test_df = df[1366:1457]

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
def compile_and_fit(model, window, patience=3, max_epochs=3):
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
resultados_df = pd.DataFrame()

# Asignar los valores de predictions a una columna llamada 'Predicciones'
resultados_df['Predicciones'] = predictions.flatten()  # Flatten para convertirlo en una sola columna

# Puedes imprimir o trabajar con resultados_df según sea necesario
print(resultados_df)

# Invertir el escalado de la columna 'Predicciones'
resultados_df['Predicciones'] = scaler.inverse_transform(resultados_df[['Predicciones']])

# Ahora, resultados_df contiene los resultados de la predicción en la escala original
print(resultados_df)