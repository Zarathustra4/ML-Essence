from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    @abstractmethod
    def forward_prop(self, x):
        """
        Returns predicted data
        :param x: numpy.ndarray - training data
        :return: numpy.ndarray - predicted data
        """
        ...

    @abstractmethod
    def back_prop(self, x, y, prediction, loss, learning_rate):
        """
        Optimizes model's parameters
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param prediction: np.ndarray - predicted value
        :param loss: LossFunctionsEnum - enum of loss functions
        :param learning_rate: float
        :return: (dw, db, loss_value)
        """
        ...

    @abstractmethod
    def fit(self, x, y, epochs, loss, learning_rate, validation_part, validation_type, scalars):
        """
        Fits model to the train data
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param epochs: number of epochs
        :param loss: LossFunctionsEnum - enum of loss functions
        :param learning_rate: float
        :param validation_part: float
        :param validation_type: ds.ValTypeEnum
        :param scalars tuple of scalars
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


