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
                        prediction: np.ndarray,
                        linear_part: np.ndarray):
        """
        :param x: model input
        :param y: model output
        :param prediction: model prediction
        :param linear_part: (w * x + b)
        :return: value of partial derivatives of loss function by W and b
        """
        ...


class MSE(LossFunction):
    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        return np.square(prediction - y).mean(axis=0)[0]

    def gradient_values(self, x: np.ndarray, y: np.ndarray, prediction: np.ndarray, f=None):
        m = x.shape[1]
        diff = prediction - y

        dw = np.sum(diff * x, axis=0) * (2 / m)
        db = np.sum(diff, axis=0) * (2 / m)
        return dw[0].T, db[0]


class CrossEntropy(LossFunction):
    def __init__(self):
        self.sigmoid = Sigmoid()

    def __numerator(self, y: np.ndarray, prediction: np.ndarray, linear_part: np.ndarray):
        numerator_matrix = (y - prediction) * self.sigmoid(linear_part) * self.sigmoid.derivative(linear_part)
        return np.sum(numerator_matrix)

    def __denominator(self, prediction: np.ndarray):
        return np.sum(prediction - np.square(prediction))

    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        m = prediction.shape[0]
        return -1 / m * np.sum(y * np.log(prediction) + (1 - y) * np.log(1 - prediction), axis=1, keepdims=True)

    def gradient_values(self, x: np.ndarray,
                        y: np.ndarray,
                        prediction: np.ndarray,
                        linear_part: np.ndarray):
        train_size = x.shape[0]
        dw = np.dot(x.T, (prediction - y)) * (1 / train_size)
        db = np.sum(prediction - y) / train_size
        return dw, db

class LossFunctionsEnum(Enum):
    MEAN_SQUARED_ERROR = MSE()
    CROSS_ENTROPY = CrossEntropy()
