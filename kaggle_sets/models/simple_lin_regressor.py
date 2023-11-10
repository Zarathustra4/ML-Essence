import numpy as np

from models.input_validator import validate_input
from models.loss_functions import LossFunctionsEnum, LossFunction
from exceptions.exceptions import ModelParameterError
from models.model import Model
import models.datasplits as ds
from plot import graph_plot


class SimpleLinRegressor(Model):
    def __init__(self, units):
        self.w = np.random.randn(units, 1)
        # self.w = np.zeros((units, 1), float)
        self.b = 1
        self.loss_functions = (LossFunctionsEnum.MEAN_SQUARED_ERROR,)  # tuple of allowed loss functions
        self.history = {}
        self._val_types = {"regular": ds.RegularValidation(), "cross_val": ...}

    def forward_prop(self, x):
        if x.shape[1] != self.w.shape[0]:
            raise ModelParameterError(
                f"Shape of x input ({x.shape}) isn't supported by the model. Has to be (m, {self.w.shape[0]})"
            )
        return x @ self.w + self.b

    def back_prop(self, x: np.ndarray,
                  y: np.ndarray,
                  prediction: np.ndarray,
                  loss: LossFunctionsEnum,
                  learning_rate=0.001):
        if loss not in self.loss_functions:
            raise ModelParameterError(
                f"Wrong loss function is passed. Linear regressor supports only these - {self.loss_functions}"
            )
        if y.shape[1] != 1:
            raise ModelParameterError(
                f"Shape of y ({y.shape}) is not supported by the model. Has to be ({x.shape[0]}, 1))"
            )

        loss_function: LossFunction = loss.value
        loss_value = loss_function(y, prediction)

        weight_lr = learning_rate
        bias_lr = learning_rate * 1e4  # bias requires larger learning rate

        dw, db = loss_function.gradient_values(x, y, prediction)

        return dw * weight_lr, db * bias_lr, loss_value

    def _validate(self, x_valid: np.ndarray, y_valid: np.ndarray, loss: LossFunction):
        val_prediction = self.forward_prop(x_valid)
        return loss(val_prediction, y_valid)

    def print_fit_progress(self, epoch: int, loss_name: str):
        print(f"[Epoch {epoch}]", end="\t")
        print(f"[loss ({loss_name}) - {self.history['loss'][epoch - 1]}]\t")
        print(f"[val_loss ({loss_name}) - {self.history['val_loss'][epoch - 1]}]\n")

    @validate_input
    def fit(self, x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            loss=LossFunctionsEnum.MEAN_SQUARED_ERROR,
            learning_rate=0.001,
            validation_part=0.2,
            validation_type="regular"):

        self.history["loss"] = [0] * epochs
        self.history["val_loss"] = [0] * epochs
        data_split_func: ds.DataSplitter = self._val_types[validation_type]

        for x_train, x_valid, y_train, y_valid, epoch in data_split_func(x, y, validation_part, epochs):
            train_prediction = self.forward_prop(x_train)
            dw, db, train_loss_value = self.back_prop(x_train, y_train, train_prediction, loss, learning_rate)
            val_loss_value = self._validate(x_valid, y_valid, loss.value) if x_valid.shape[0] != 0 else 0
            self.w -= dw
            self.b -= db
            self.history["loss"][epoch - 1] = train_loss_value
            self.history["val_loss"][epoch - 1] = val_loss_value
            self.print_fit_progress(epoch, loss.name)

        return self.history

    def predict(self, x: np.ndarray):
        return self.forward_prop(x)