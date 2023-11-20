from functools import wraps

from exceptions.exceptions import ModelParameterError
import models.datasplits as ds
import numpy as np

from models.loss_functions import LossFunctionsEnum
import models.data_scalar as scal


def validate_input(func):
    @wraps(func)
    def wrapper(self, x: np.ndarray,
                y: np.ndarray,
                epochs: int,
                loss=LossFunctionsEnum.MEAN_SQUARED_ERROR,
                validation_part=0.2,
                validation_type=ds.ValDataSplitEnum.REGULAR_VAL,
                scalars: tuple[scal.DataScalar] = None):

        if x is not None and x.shape[1] != self.w.shape[0]:
            raise ModelParameterError(
                f"Shape of x input ({x.shape}) isn't supported by the model. Has to be (m, {self.w.shape[0]})"
            )
        if y is not None and y.shape[1] != 1:
            raise ModelParameterError(
                f"Shape of y ({y.shape}) is not supported by the model. Has to be ({x.shape[0]}, 1))"
            )
        if validation_part > 1 or validation_part < 0:
            raise ModelParameterError(
                f"Validation part can not be more than 1 or less than 0"
            )
        if validation_type not in self._val_types:
            raise ModelParameterError(
                f"Impossible value for validation_type - {validation_type}. Possible values - {self.__val_types}"
            )

        return func(self, x, y, epochs, loss, validation_part, validation_type, scalars)

    return wrapper
