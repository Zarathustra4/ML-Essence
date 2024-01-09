import json
from abc import abstractmethod

import numpy as np

from kaggle_sets.custom.preprocessing.data_scalar import create_data_scalar


class Model:

    @abstractmethod
    def forward_prop(self, x):
        """
        Returns predicted data
        :param x: numpy.ndarray - training data
        :return: numpy.ndarray - predicted data
        """
        ...

    @abstractmethod
    def back_prop(self, x, y, prediction, loss_enum, learning_rate):
        """
        Optimizes model's parameters
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param prediction: np.ndarray - predicted value
        :param loss_enum: LossEnum - enum of loss functions
        :param learning_rate: float
        :return: (dw, db, loss_value)
        """
        ...

    @abstractmethod
    def fit(self, x, y, epochs, validation_part, validation_splitter):
        """
        Fits model to the train data
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param epochs: number of epochs
        :param validation_part: float
        :param validation_splitter: DataSplitter - object, which split data onto train and validation
        :return: dict - history of training
        """
        ...

    @abstractmethod
    def predict(self, x):
        """
        Predicts the value
        :param x: numpy.ndarray - training data
        :return: numpy.ndarray - predicted data
        """
        ...

    @abstractmethod
    def update_parameters(self, dw: np.ndarray, db: float):
        """
        Optimizes the parameters of model
        :param dw: derivative by weights
        :param db: derivative by bias
        :return:
        """

    @abstractmethod
    def save(self, path: str):
        """
        Save model to a given location
        :param path: file location
        :return:
        """

    @staticmethod
    def _get_history(metrics: tuple, epochs: int) -> dict:
        history = {"loss": [0] * epochs, "val_loss": [0] * epochs}

        metric_names = [metric.name for metric in metrics]
        history.update({f"{metric_name}": [0] * epochs for metric_name in metric_names})
        history.update({f"val_{metric_name}": [0] * epochs for metric_name in metric_names})

        return history

    def update_history(self, history: dict, epoch: int, train_loss_value, val_loss_value, metrics, y_train, x_train,
                       y_valid, x_valid):
        history["loss"][epoch - 1] = train_loss_value
        history["val_loss"][epoch - 1] = val_loss_value

        for metric in metrics:
            train_metric_value = metric(y_train, self.predict(x_train))
            val_metric_value = metric(y_valid, self.predict(x_valid))
            metric_name = metric.name

            history[f"{metric_name}"][epoch - 1] = train_metric_value
            history[f"val_{metric_name}"][epoch - 1] = val_metric_value

        return history

    def load(self, path: str):
        """
        Loads model from a given file location
        :return: None
        """
        with open(path, 'r') as file:
            model_data = json.load(file)
            self.w = np.array(model_data["w"])
            self.b = model_data["b"]

            self.scalars = []
            for json_scalar in model_data["scalars"]:
                self.scalars.append(
                    create_data_scalar(json_scalar)
                )
