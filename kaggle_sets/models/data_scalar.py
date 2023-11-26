import numpy as np
from abc import abstractmethod, ABC


class DataScalar(ABC):

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
        self._std_values = array.std(axis=0)

    def _scale(self, array: np.ndarray):
        return (array - self._mean_values) / self._std_values

