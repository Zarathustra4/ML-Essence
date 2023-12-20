from enum import Enum
import numpy as np


class Metric:
    def __init__(self, name: str):
        self.name: str = name

    def _metric(self, prediction: np.ndarray, y: np.ndarray):
        """
        :return: value of a metric
        """
    
    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        return self._metric(prediction, y)


class MSE(Metric):
    def __init__(self):
        super().__init__("mean_squared_error")

    """ Mean Squared Error Metric"""
    def _metric(self, prediction: np.ndarray, y: np.ndarray):
        diff = prediction - y
        m = diff.shape[1]
        return (1 / m) * (diff @ diff.T)[0, 0]


class MAE(Metric):
    def __init__(self):
        super().__init__("mean_absolute_error")

    """ Mean Absolute Error Metric"""
    def _metric(self, prediction: np.ndarray, y: np.ndarray):
        m = y.shape[0]
        diff = np.abs(prediction - y)
        return np.sum(diff, axis=0) * (1 / m) # <= there's no axis in np.abs =)


class Accuracy(Metric):
    def __init__(self):
        super().__init__("accuracy")

    """ Accuracy Metric"""
    def _metric(self, prediction: np.ndarray, y: np.ndarray):
        correct_predictions = np.sum(prediction.round() == y)
        total_predictions = len(y)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy


class R2(Metric):
    def __init__(self):
        super().__init__("r_squared")

    """ R Squared Metric """
    def _metric(self, prediction: np.ndarray, y: np.ndarray):
        SS_res = np.sum(np.square(y - prediction))
        SS_tot = np.sum(np.square(y - np.mean(y)))
        r2 = 1 - SS_res / (SS_tot + np.finfo(float).eps)
        return r2


class MetricsEnum(Enum):
    MEAN_SQUARED_ERROR = MSE()
    MEAN_ABSOLUTE_ERROR = MAE()
    ACCURACY = Accuracy()
    R_SQUARED = R2()
