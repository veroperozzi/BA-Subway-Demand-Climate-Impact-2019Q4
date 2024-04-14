import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

print(tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)

# index_col=0 Cual columna del CSV se utiliza como indice del DataFrame resultante.
train_df = pd.read_csv(
    '/home/vero/Documentos/proyectos_python/thesis_subway_climatic_impact1/Variables_Exogenas/Deep_Learning/GRU/GRU_3VEx/train.csv',
    index_col=0)
test_df = pd.read_csv(
    '/home/vero/Documentos/proyectos_python/thesis_subway_climatic_impact1/Variables_Exogenas/Deep_Learning/GRU/GRU_3VEx/test.csv',
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
class SymmetricMeanAbsolutePercentageError(tf.keras.metrics.Metric):
    def __init__(self, name='smape', **kwargs):
        super(SymmetricMeanAbsolutePercentageError, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + tf.math.abs(y_pred) + 1e-7)
        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

####################################
def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='loss', patience=patience, mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[SymmetricMeanAbsolutePercentageError()])

    history = model.fit(window.train,
                        epochs=max_epochs,
                        callbacks=[early_stopping])

    return history


# Definir las columnas que serán utilizadas como entrada y salida
input_columns = ['TOTAL', 'PRECIPITACION', 'FERIADO', 'LABORABLE', 'day_sin', 'day_cos']
output_columns = ['TOTAL']

multi_window = DataWindow(input_width=90, label_width=90, shift=1, label_columns=output_columns)

GRU_model = Sequential([
    Concatenate(axis=-1), # Combinar secuencia de serie de tiempo y secuencia de variables exógenas
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

# Visualizar la predicción
multi_window.plot(model=GRU_model)
plt.suptitle('Predicción GRU', y=0.95, fontsize=16)
plt.show()
