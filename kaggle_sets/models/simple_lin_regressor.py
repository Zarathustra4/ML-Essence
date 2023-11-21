import random

import numpy as np

from models.input_validator import validate_input
from models.loss_functions import LossFunctionsEnum, LossFunction
from exceptions.exceptions import ModelParameterError
from models.model import Model
import models.datasplits as ds
from models.optimizers import SGD, GradientDescent, Optimizer
from plot import graph_plot
import models.data_scalar as scal


class SimpleLinRegressor(Model):
    def __init__(self, units, optimizer=SGD()):
        self.w = np.random.randn(units, 1)
        self.b = 1
        self.loss_functions = (LossFunctionsEnum.MEAN_SQUARED_ERROR,)  # tuple of allowed loss functions
        self.history = {}
        self._val_types = {ds.ValDataSplitEnum.REGULAR_VAL: ds.RegularValidation(),
                           ds.ValDataSplitEnum.CROSS_VAL: ds.CrossValidation()}
        self.scalars: list[scal.DataScalar] = []
        self.optimizer: Optimizer = optimizer

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

        dw, db = loss_function.gradient_values(x, y, prediction)

        return dw * learning_rate, db * learning_rate, loss_value

    def _validate(self, x_valid: np.ndarray, y_valid: np.ndarray, loss: LossFunction):
        val_prediction = self.forward_prop(x_valid)
        return loss(val_prediction, y_valid)

    def print_fit_progress(self, epoch: int, loss_name: str):
        print(f"[Epoch {epoch}]", end="\t")
        print(f"[loss ({loss_name}) - {self.history['loss'][epoch - 1]}]\t")
        print(f"[val_loss ({loss_name}) - {self.history['val_loss'][epoch - 1]}]\n")

    def set_scale_data(self, data: np.ndarray):
        data = np.array(data)
        for i in range(len(self.scalars)):
            self.scalars[i].set_values(data)
            data = self.scalars[i](data)

    def _scale_data(self, data: np.ndarray):
        for scalar in self.scalars:
            data = scalar(data)

        return data

    def update_parameters(self, dw: np.ndarray, db: float):
        self.w -= dw
        self.b -= db

    @validate_input
    def fit(self, x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            loss=LossFunctionsEnum.MEAN_SQUARED_ERROR,
            validation_part=0.2,
            validation_type=ds.ValDataSplitEnum.REGULAR_VAL,
            scalars: tuple[scal.DataScalar] = None):

        self.scalars = list(scalars) if scalars else []
        self.set_scale_data(x)
        x = self._scale_data(x)

        self.history["loss"] = [0] * epochs
        self.history["val_loss"] = [0] * epochs
        data_split_func: ds.DataSplitter = self._val_types[validation_type]

        for x_train, x_valid, y_train, y_valid, epoch in data_split_func(x, y, validation_part, epochs):
            train_loss_value = self.optimizer.optimize(x_train, y_train, self)
            val_loss_value = self._validate(x_valid, y_valid, loss.value)

            self.history["loss"][epoch - 1] = train_loss_value
            self.history["val_loss"][epoch - 1] = val_loss_value

            self.print_fit_progress(epoch, loss.name)

        return self.history

    def predict(self, x: np.ndarray):
        x = self._scale_data(x)
        return self.forward_prop(x)


def f(x):
    return x[0] + 2 * x[1] - 3 * x[2]


if __name__ == "__main__":
    x = [random.sample(range(-20, 20), 3) for i in range(3000)]
    y = [f(num) for num in x]

    x = np.array(x)
    y = np.array(y, ndmin=2).T

    model = SimpleLinRegressor(units=3)

    history = model.fit(x, y, epochs=1000,
                        loss=LossFunctionsEnum.MEAN_SQUARED_ERROR,
                        learning_rate=1e-6,
                        scalars=(scal.Normalizer(), scal.Standardizer()))

    graph_plot.plot_loss_history(history)

    from random import randint

    for i in range(10):
        test_x = np.array([randint(0, 10), randint(0, 10), randint(0, 10)], ndmin=2)

        true_result = f(test_x[0])
        prediction = model.predict(test_x)

        print(f"prediction - {prediction[0, 0]} | true result - {true_result}")
        print("-" * 20)
