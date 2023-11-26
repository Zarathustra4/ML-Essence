from enum import Enum
from numpy import (
    ndarray,
    abs as np_abs,
    sum as np_sum,
    mean as np_mean
)


class Metric:
    def _metric(self, prediction: ndarray, y: ndarray):
        """
        :return: value of a metric
        """
    
    def __call__(self, prediction: ndarray, y: ndarray):
        return self._metric(prediction, y)


class MSE(Metric):
    """ Mean Squared Error Metric"""
    def _metric(self, prediction: ndarray, y: ndarray):
        diff = prediction - y
        m = diff.shape[1]
        return (1 / m) * (diff @ diff.T)[0, 0]


class MAE(Metric):
    """ Mean Absolute Error Metric"""
    def _metric(self, prediction: ndarray, y: ndarray):
        m = y.shape[0]
        diff = np_abs(prediction - y)
        return np_sum(diff, axis=0) * (1 / m)


class Accuracy(Metric):
    """ Accuracy Metric"""
    def _metric(self, prediction: ndarray, y: ndarray):
        correct_predictions = np_sum(prediction == y)
        total_predictions = len(y)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy


class R2(Metric):
    """ R Squared Metric """
    def _metric(self, prediction: ndarray, y: ndarray):
        mean_y = np_mean(y)
        ss_total = np_sum((y - mean_y) ** 2)
        ss_residual = np_sum((y - prediction) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
        return r2


class MetricsEnum(Enum):
    MEAN_SQUARED_ERROR = MSE()
    MEAN_ABSOLUTE_ERROR = MAE()
    ACCURACY = Accuracy()
    R_SQUARED = R2()