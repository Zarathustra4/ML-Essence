import numpy as np
from abc import abstractmethod, ABC


class DataScalar(ABC):

    def __init__(self, array=np.array([])):
        if array:
            self.fit(array)

    @abstractmethod
    def _scale(self, array: np.ndarray):
        """
        Scales data
        :param array: array of data
        :return: scaled array
        """

    @abstractmethod
    def set_values(self, **values):
        """
        Sets data scalar values
        """

    @abstractmethod
    def fit(self, array: np.ndarray):
        """
        Sets values for scaling
        :param array: data
        :return: None
        """

    @abstractmethod
    def data(self) -> dict:
        """
        Returns scalar data
        :return: dict
        """

    def __call__(self, array: np.ndarray):
        return self._scale(array)


class Normalizer(DataScalar):
    def __init__(self, array=np.array([])):
        self._min_values = 0
        self._max_values = 1
        super().__init__(array)

    def set_values(self, **values):
        min_values = values.get("min_values")
        max_values = values.get("max_values")
        if min_values is None:
            raise ValueError("min_values parameter is required")
        if max_values is None:
            raise ValueError("max_values parameter is required")

        self._min_values = min_values
        self._max_values = max_values

    def fit(self, array: np.ndarray):
        self._min_values = array.min(axis=0)
        self._max_values = array.max(axis=0)

    def _scale(self, array: np.ndarray):
        return (array - self._min_values) / (self._max_values - self._min_values)

    def data(self) -> dict:
        min_values = self._min_values
        max_values = self._max_values
        if isinstance(min_values, np.ndarray):
            min_values = min_values.tolist()
        else:
            min_values = float(min_values)

        if isinstance(max_values, np.ndarray):
            max_values = max_values.tolist()
        else:
            min_values = float(max_values)

        return {"type": "normalizer", "_min_values": min_values, "_max_values": max_values}


class Standardizer(DataScalar):
    def __init__(self, array=np.array([])):
        self._mean_values = 0
        self._std_values = 1
        super().__init__(array)

    def set_values(self, **values):
        mean_values = values.get("mean_values")
        std_values = values.get("std_values")

        if mean_values is None:
            raise ValueError("mean_values parameter is required")
        if std_values is None:
            raise ValueError("std_values parameter is required")

        self._mean_values = mean_values
        self._std_values = std_values

    def fit(self, array: np.ndarray):
        self._mean_values = array.mean(axis=0)
        self._std_values = array.std(axis=0)

    def _scale(self, array: np.ndarray):
        return (array - self._mean_values) / self._std_values

    def data(self) -> dict:
        mean_values = self._mean_values
        std_values = self._std_values
        if isinstance(mean_values, np.ndarray):
            mean_values = mean_values.tolist()
        else:
            std_values = float(mean_values)

        if isinstance(std_values, np.ndarray):
            std_values = std_values.tolist()
        else:
            std_values = float(std_values)

        return {"type": "standardizer", "_mean_values": mean_values, "_std_values": std_values}


def create_data_scalar(scalar_dict: dict):
    if scalar_dict["type"] == "normalizer":
        scalar = Normalizer()
        min_values = np.array(scalar_dict["_min_values"])
        max_values = np.array(scalar_dict["_max_values"])
        scalar.set_values(min_values=min_values, max_values=max_values)
        return scalar

    if scalar_dict["type"] == "standardizer":
        scalar = Standardizer()
        mean_values = np.array(scalar_dict["_mean_values"])
        std_values = np.array(scalar_dict["_std_values"])
        scalar.set_values(mean_values=mean_values, std_values=std_values)
        return scalar
