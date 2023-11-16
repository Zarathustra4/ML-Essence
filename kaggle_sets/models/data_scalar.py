import numpy as np
from abc import abstractmethod


class DataScalar:

    def __init__(self, array=np.array([])):
        if array:
            self.set_values(array)

    @abstractmethod
    def _scale(self, array: np.ndarray):
        """
        Scales data
        :param array: array of data
        :return: scaled array
        """
        ...

    @abstractmethod
    def set_values(self, array: np.ndarray):
        ...

    def __call__(self, array: np.ndarray):
        return self._scale(array)


class Normalizer(DataScalar):
    def __init__(self, array=np.array([])):
        self._min_values = 0
        self._max_values = 1
        super().__init__(array)

    def set_values(self, array: np.ndarray):
        self._min_values = array.min(axis=0)
        self._max_values = array.max(axis=0)

    def _scale(self, array: np.ndarray):
        return (array - self._min_values) / (self._max_values - self._min_values)


class Standardizer(DataScalar):
    def __init__(self, array=np.array([])):
        self._mean_values = 0
        self._std_values = 1
        super().__init__(array)

    def set_values(self, array: np.ndarray):
        self._mean_values = array.mean(axis=0)
        self._std_values = array.mean(axis=0)

    def _scale(self, array: np.ndarray):
        return (array - self._mean_values) / self._std_values


def standardize(array: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = array.mean(axis=0)
    if std is None:
        std = array.std(axis=0)
    return (array - mean) / std, mean, std


def scale(array: np.ndarray):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def unstandardize(std_array: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return mean + std_array * std


if __name__ == "__main__":
    x = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [1, 5, 6],
                  [9, 9, 9],
                  [1, 2, 3]])

    x = scale(x)
    print(x)
