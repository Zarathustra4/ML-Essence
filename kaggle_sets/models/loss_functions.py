from abc import abstractmethod

import numpy as np
from enum import Enum

from models.activations import Sigmoid


class LossFunction:
    @abstractmethod
    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        """
        :return: value of loss function
        """
        ...

    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        return self.loss_function(prediction, y)

    @abstractmethod
    def gradient_values(self, x: np.ndarray,
                        y: np.ndarray,
                        prediction: np.ndarray):
        """
        :param x: model input
        :param y: model output
        :param prediction: model prediction
        :return: value of partial derivatives of loss function by W and b
        """
        ...


class MSE(LossFunction):
    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        return np.square(prediction - y).mean(axis=0)[0]

    def gradient_values(self, x: np.ndarray, y: np.ndarray, prediction: np.ndarray):
        m = x.shape[1]
        diff = prediction - y

        dw = np.sum(diff * x, axis=0) * (2 / m)
        db = np.sum(diff, axis=0) * (2 / m)
        return dw[0].T, db[0]


class CrossEntropy(LossFunction):
    def __init__(self):
        self.sigmoid = Sigmoid()

    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        m = prediction.shape[0] or 1  # to avoid zero division
        summation_matrix = y * np.log(prediction) + (1 - y) * np.log(1 - prediction)
        return -1 / m * np.sum(summation_matrix, axis=0)

    def gradient_values(self, x: np.ndarray,
                        y: np.ndarray,
                        prediction: np.ndarray):
        train_size = x.shape[0]
        dw = np.dot(x.T, (prediction - y)) * (1 / train_size)
        db = np.sum(prediction - y) / train_size
        return dw, db


class LossEnum(Enum):
    MEAN_SQUARED_ERROR = MSE()
    CROSS_ENTROPY = CrossEntropy()
