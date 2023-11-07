import numpy as np
import pandas as pd
from models.loss_functions import LossFunctionsEnum, LossFunction
from exceptions.exceptions import ModelParameterError
from model import Model
from plot import graph_plot


class SimpleLinRegressor(Model):
    def __init__(self, units):
        self.w = np.random.rand(units, 1)
        self.b = 0
        self.loss_functions = (LossFunctionsEnum.MEAN_SQUARED_ERROR,)  # tuple of allowed loss functions
        self.history = {}

    def _split_data(self, x: np.ndarray, y: np.ndarray, validation_part: float):
        data_size = x.shape[0]
        train_size = int(data_size * 1 - validation_part)

        data = np.concatenate((x, y), axis=1)
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1:]

        x_train = x[:train_size, :]
        x_valid = x[train_size:, :]
        y_train = y[:train_size, :]
        y_valid = y[train_size:, :]

        return x_train, x_valid, y_train, y_valid

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

    def fit(self, x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            loss: LossFunctionsEnum,
            learning_rate=0.001,
            validation_part=0.2):
        # TODO: write the decorator for the validations
        if x.shape[1] != self.w.shape[0]:
            raise ModelParameterError(
                f"Shape of x input ({x.shape}) isn't supported by the model. Has to be (m, {self.w.shape[0]})"
            )
        if y.shape[1] != 1:
            raise ModelParameterError(
                f"Shape of y ({y.shape}) is not supported by the model. Has to be ({x.shape[0]}, 1))"
            )
        if validation_part > 1 or validation_part < 0:
            raise ModelParameterError(
                f"Validation part can not be more than 1 or less than 0"
            )

        x_train, x_valid, y_train, y_valid = self._split_data(x, y, validation_part)

        self.history["loss"] = [0] * epochs
        self.history["val_loss"] = [0] * epochs

        for epoch in range(1, epochs):
            train_prediction = self.forward_prop(x_train)
            dw, db, train_loss_value = self.back_prop(x_train, y_train, train_prediction, loss, learning_rate)
            val_loss_value = self._validate(x_valid, y_valid, loss.value)
            self.w -= dw
            self.b -= db
            self.history["loss"][epoch - 1] = train_loss_value
            self.history["val_loss"][epoch - 1] = val_loss_value
            self.print_fit_progress(epoch, loss.name)

        return self.history

    def predict(self, x: np.ndarray):
        return self.forward_prop(x)


if __name__ == "__main__":
    x_test = [[i, i / 2, i / 3] for i in range(100)]
    y_test = [num[0] + 2 * num[1] - 1 for num in x_test]

    x_test = np.array(x_test)
    y_test = np.array(y_test, ndmin=2).T

    model = SimpleLinRegressor(units=3)

    history = model.fit(x_test, y_test, epochs=400, loss=LossFunctionsEnum.MEAN_SQUARED_ERROR, learning_rate=1e-7)
    graph_plot.plot_loss_history(history)
