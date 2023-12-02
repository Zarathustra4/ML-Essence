from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray):
        ...

    @abstractmethod
    def derivative(self, x: np.ndarray):
        ...


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray):
        exp_sum = np.sum(np.exp(x))
        return np.exp(x) / exp_sum

    def derivative(self, x: np.ndarray):
        #TODO
        pass


class ActivationsEnum(Enum):
    SIGMOID = Sigmoid()
    SOFTMAX = Softmax()
