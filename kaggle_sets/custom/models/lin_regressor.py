import numpy as np

from custom.functions.loss_functions import LossEnum, LossFunction
from exceptions.exceptions import ModelParameterError
from custom.models.model import Model
import custom.preprocessing.datasplits as ds
from custom.models.optimizers import SGD, Optimizer
import custom.preprocessing.data_scalar as scal
import json


class LinRegressor(Model):
    def __init__(self, units: int, optimizer=SGD(loss_enum=LossEnum.MEAN_SQUARED_ERROR), data_scalars: tuple = ()):
        self.w = np.random.randn(units, 1)
        self.b = 1
        self.loss_functions = (LossEnum.MEAN_SQUARED_ERROR,)  # tuple of allowed loss functions
        self.history = {}
        self.optimizer: Optimizer = optimizer
        self.scalars: list[scal.DataScalar] = list(data_scalars)

    def forward_prop(self, x):
        if x.shape[1] != self.w.shape[0]:
            raise ModelParameterError(
                f"Shape of x input ({x.shape}) isn't supported by the model. Has to be (m, {self.w.shape[0]})"
            )
        return x @ self.w + self.b

    def back_prop(self, x: np.ndarray,
                  y: np.ndarray,
                  prediction: np.ndarray,
                  loss_enum: LossEnum,
                  learning_rate=0.001):
        if loss_enum not in self.loss_functions:
            raise ModelParameterError(
                f"Wrong loss function is passed. Linear regressor supports only these - {self.loss_functions}"
            )
        if y.shape[1] != 1:
            raise ModelParameterError(
                f"Shape of y ({y.shape}) is not supported by the model. Has to be ({x.shape[0]}, 1))"
            )

        loss_function: LossFunction = loss_enum.value
        loss_value = loss_function(y, prediction)

        dw, db = loss_function.gradient_values(x, y, prediction)

        return dw * learning_rate, db * learning_rate, loss_value

    def _validate(self, x_valid: np.ndarray, y_valid: np.ndarray, loss: LossFunction):
        val_prediction = self.forward_prop(x_valid)
        return loss(val_prediction, y_valid)

    def print_fit_progress(self, epoch: int, loss_name: str):
        print(f"{'.'*21} Epoch {epoch} {'.'*21}")
        print(f": {'Training Loss   ('+loss_name+')':<25} : {self.history['loss'][epoch - 1]:.4f} :")
        print(f": {'Validation Loss ('+loss_name+')':<25} : {self.history['val_loss'][epoch - 1]:.4f} :")
        print('.' * 53 + '\n')

    def set_scale_data(self, data: np.ndarray):
        data = np.array(data)
        for i in range(len(self.scalars)):
            self.scalars[i].fit(data)
            data = self.scalars[i](data)

    def _scale_data(self, data: np.ndarray):
        for scalar in self.scalars:
            data = scalar(data)

        return data

    def update_parameters(self, dw: np.ndarray, db: float):
        self.w -= dw
        self.b -= db

    def fit(self, x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            validation_part: float = 0.2,
            validation_splitter: ds.DataSplitter = ds.RegularValidation(),
            metrics: tuple = ()):

        loss = self.optimizer.get_loss_enum()

        self.set_scale_data(x)
        x = self._scale_data(x)

        self.history = LinRegressor._get_history(metrics, epochs)

        for x_train, x_valid, y_train, y_valid, epoch in validation_splitter(x, y, validation_part, epochs):
            train_loss_value = self.optimizer.optimize(x_train, y_train, self)
            val_loss_value = self._validate(x_valid, y_valid, loss.value)

            self.update_history(self.history, epoch, train_loss_value, val_loss_value,
                                metrics, y_train, x_train, y_valid, x_valid)
            
            self.print_fit_progress(epoch, loss.name)

        return self.history

    def predict(self, x: np.ndarray):
        x = self._scale_data(x)
        return self.forward_prop(x)

    def save(self, path: str):
        model_data = {
            "w": self.w.tolist(),
            "b": self.b,
            "scalars": [scalar.data() for scalar in self.scalars]
        }
        with open(path, 'w') as file:
            json.dump(model_data, file)


