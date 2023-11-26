from abc import abstractmethod

import numpy as np
from enum import Enum


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
    def gradient_values(self, x: np.ndarray, y: np.ndarray, prediction):
        """
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


class LossEnum(Enum):
    MEAN_SQUARED_ERROR = MSE()
