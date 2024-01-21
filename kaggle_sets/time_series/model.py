import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config as conf
from pathlib import Path
from plot import graph_plot

class Forecaster:
    def __init__(self):
        self.window_size = conf.TS_WINDOW_SIZE
        self.path = conf.TS_MODEL_PATH
        self.model: tf.keras.Model = self.__init_model()

    def __init_model(self):
        if Path(self.path).exists():
            return tf.keras.models.load_model(self.path)

        return Forecaster.get_untrained_model()

    @staticmethod
    def get_untrained_model(window_size: int = conf.TS_WINDOW_SIZE):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                   strides=1,
                                   activation="relu",
                                   padding='causal',
                                   input_shape=[window_size, 1]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=tf.keras.optimizers.Adam(lr=5e-3),
                      metrics=["mae"])

        return model

    def reset_model(self):
        self.model = Forecaster.get_untrained_model()

    def fit(self, train_set, epochs=50, plot_history=True):
        history = self.model.fit(train_set, epochs=epochs)

        self.model.save(conf.TS_MODEL_PATH)

        if plot_history:
            plt.plot(history.history["loss"])
            plt.show()

        return history

    def forecast(self, series, n_steps: int = 100, plot_forecast=True, zoom=True):
        x = series[-self.window_size:]
        x = x.reshape(1, self.window_size, 1)

        predictions = []

        for _ in range(n_steps):
            next_step = self.model.predict(x)
            x = np.concatenate([x[:, 1:, :], next_step.reshape(1, 1, 1)], axis=1)
            predictions.append(next_step[0, 0])

        if plot_forecast:
            graph_plot.plot_forecast(series, predictions, zoom)

        return predictions
