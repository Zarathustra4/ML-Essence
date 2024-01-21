import numpy as np
import pandas as pd

from data_preparation.dataset_to_numpy import DatasetToNumpy
from plot.graph_plot import plot_loss_history, plot_metric_history
from custom.functions.loss_functions import LossEnum
from custom.functions.metrics import MetricsEnum, MSE, MAE, R2
from custom.models.lin_regressor import LinRegressor
from custom.models.optimizers import SGD
from custom.preprocessing import data_scalar as scal
import config as conf
from pathlib import Path

from custom.preprocessing.datasplits import CrossValidation


class RegressionService:
    def __init__(self):
        self.numpy_caster = DatasetToNumpy("mosquito-indicator", csv_delimeter=",")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.numpy_caster(["date"],
                                                                                     y_column="mosquito_Indicator")
        self.path = conf.LIN_REGRESSOR_PATH
        self.model: LinRegressor = self._get_model()

    def train_model(
            self,
            epochs: int = 150,
            validation_split: float = 0.2,
            metrics=(MetricsEnum.MEAN_SQUARED_ERROR.value, MetricsEnum.MEAN_ABSOLUTE_ERROR.value),
            plot_history=True
    ):
        """
        Trains a model
        :param epochs: number of epochs
        :param validation_split: validation part of dataset
        :param metrics: calculated metrics
        :param plot_history: set True if you want to plot training history
        :return: dict - training history
        """
        history = self.model.fit(self.x_train, self.y_train, epochs, validation_split, metrics=metrics)

        if plot_history:
            plot_loss_history(history)
            plot_metric_history(history, "mean_absolute_error")
            plot_metric_history(history, "mean_squared_error")

        self.model.save(self.path)

        return history

    @staticmethod
    def get_untrained_model():
        return LinRegressor(units=4,
                            data_scalars=(scal.Normalizer(),),
                            optimizer=SGD(
                                loss_enum=LossEnum.MEAN_SQUARED_ERROR,
                                batch_size=128, learning_rate=1e-3
                            ))

    def _get_model(self):
        model = RegressionService.get_untrained_model()
        if Path(self.path).exists():
            model.load(self.path)

        return model

    def reset_model(self) -> None:
        """
            Creates an untrained model
            :return: None
        """
        self.model = RegressionService.get_untrained_model()

    def create_train_model(
            self,
            epochs: int = 150,
            validation_split: float = 0.2,
            metrics=(MetricsEnum.MEAN_SQUARED_ERROR.value, MetricsEnum.MEAN_ABSOLUTE_ERROR.value),
            plot_history=True
    ) -> dict:
        """
            Creates and trains a model
            :param metrics: metrics, which are calculated while model is training
            :param epochs: number of learning repetitions
            :param validation_split: float value, percent of validating data
            :param plot_history: True if user wants to plot train history, else - False
            :return: history - dict
        """

        self.model = RegressionService.get_untrained_model()

        return self.train_model(
            epochs=epochs,
            validation_split=validation_split,
            metrics=metrics,
            plot_history=plot_history
        )

    def test_model(self) -> dict:
        """
        Tests a model prediction and return a dict of calculated metrics
        :return: dict
        """
        predictions = self.model.predict(self.x_test)

        mse = MSE()
        mae = MAE()
        r2 = R2()

        return {
            "mse": mse(predictions, self.y_test),
            "mae": mae(predictions, self.y_test),
            "r2": r2(predictions, self.y_test)
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_by_csv(self, filename: str, delimeter: str = ","):
        df = pd.read_csv(filename)
        df = df.drop("date", axis=1)
        x = df.to_numpy()
        return self.model.predict(x)


def train_save_regressor():
    service = RegressionService()

    history = service.create_train_model()
    metrics = service.test_model()

    print(f"| Prediction Mean Squared Error | {metrics['mae'][0]: .2f}")
    print(f"| Prediction R Squared          | {metrics['r2']: .2f}")
    print(f"| Prediction Mean Squared Error | {metrics['mse']: .2f}")
    print(f"| Final Mean Absolute Error     | {history['mean_absolute_error'][-1][0]: .2f}")
