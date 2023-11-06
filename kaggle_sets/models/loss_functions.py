import numpy as np
from enum import Enum


class LossFunction:
    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        """
        :return: value of loss function
        """
        ...

    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        return self.loss_function(prediction, y)

    def gradient_values(self, x: np.ndarray, y: np.ndarray, prediction):
        """
        :return: value of partial derivatives of loss function by W and b
        """
        ...


class MSE(LossFunction):
    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        diff = prediction - y
        m = diff.shape[1]
        return (1 / m) * (diff @ diff.T)[0, 0]

    def gradient_values(self, x: np.ndarray, y: np.ndarray, prediction: np.ndarray):
        m = x.shape[1]
        diff = prediction - y

        dw = np.sum(diff * x, axis=0) * (2 / m)
        db = np.sum(diff, axis=0) * (2 / m)
        return dw[0], db[0]


class LossFunctions(Enum):
    MEAN_SQUARED_ERROR = MSE()
