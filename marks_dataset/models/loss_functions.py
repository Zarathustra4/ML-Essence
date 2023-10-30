import numpy as np
from enum import Enum


def fix_dims(vec: np.ndarray, axis=0):
    return np.expand_dims(vec, axis=axis) if len(vec.shape) == 1 else vec


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
        prediction = fix_dims(prediction)
        diff = prediction - y
        m = diff.shape[1]
        return (1 / m) * (diff @ diff.T)[0, 0]

    def gradient_values(self, x: np.ndarray, y: np.ndarray, prediction: np.ndarray):
        x = fix_dims(x)
        y = fix_dims(y)
        m = x.shape[1]
        diff = prediction - y

        dw = np.sum(diff * x, axis=1) * (2 / m)
        db = np.sum(diff, axis=1) * (2 / m)
        return fix_dims(dw), db


class LossFunctions(Enum):
    MEAN_SQUARED_ERROR = MSE()
