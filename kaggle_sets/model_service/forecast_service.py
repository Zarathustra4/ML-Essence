from time_series.model import Forecaster
import time_series.data_preparation as dp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import config as conf


class ForecastService:
    def __init__(self):
        self.model = Forecaster()

    def train_model(self, epochs=50):
        """
        Trains model using a dataset
        :param epochs: number of epochs
        :return: history of training
        """
        train_data = dp.get_train_windowed_data()
        history = self.model.fit(train_data, epochs)
        return history.history

    def reset_model(self):
        """
        Model become untrained
        :return: None
        """
        self.model.reset_model()

    def test_model(self, ahead_steps=conf.TS_WINDOW_SIZE, plot=True) -> dict:
        """
        Tests model by a test set
        :param ahead_steps: ahead time steps for forecasting
        :param plot: set True if you want to plot series and forecast
        :return: dict with calculated metrics
        """
        series = dp.get_test_series()
        test_series = series[:-ahead_steps]
        forecast = self.model.forecast(test_series, ahead_steps, plot_forecast=False)

        if plot:
            plt.plot(range(len(series)), series, label="Original series")
            plt.plot(range(len(test_series), len(series)), forecast, label="Forecast")
            plt.show()

        return {
            "mse": tf.keras.metrics.mean_squared_error(series[-ahead_steps:], forecast).numpy(),
            "mae": tf.keras.metrics.mean_absolute_error(series[-ahead_steps:], forecast).numpy()
        }

    def predict(self, series, ahead_steps=conf.TS_WINDOW_SIZE):
        """
        Making forecast
        :param ahead_steps: ahead time steps for forecast
        :param series: given series
        :return: forecast
        """
        return self.model.forecast(series, ahead_steps)

    def predict_by_csv(self, filename, ahead_steps=conf.TS_WINDOW_SIZE, delimeter=",", plot_forecast: bool = True):
        series = pd.read_csv(filename, delimiter=delimeter)["Temp"].to_numpy()
        return self.model.forecast(series, plot_forecast=plot_forecast, n_steps=ahead_steps)


def train_save_forecaster():
    service = ForecastService()
    service.reset_model()
    service.train_model(epochs=85)

    print("| --- Testing model --- |")
    print(service.test_model())
