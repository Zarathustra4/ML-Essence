import numpy as np

from exceptions.exceptions import ModelParameterError
from models.loss_functions import LossEnum, LossFunction
from models.model import Model
from models.optimizers import SGD, Optimizer
import models.datasplits as ds
from activations import ActivationFunction, Sigmoid
import models.data_scalar as scal
from plot.graph_plot import plot_loss_history


class BinaryClassifier(Model):
    def __init__(self, units: int, activation=Sigmoid(),
                 optimizer: Optimizer = SGD(loss_enum=LossEnum.CROSS_ENTROPY), data_scalars: tuple = ()):
        self.w: np.ndarray = np.random.randn(units, 1)
        self.b: float = 1.
        self.activation: ActivationFunction = activation
        self.loss_functions: tuple[LossEnum] = (LossEnum.CROSS_ENTROPY, )
        self.history: dict = {}
        self.scalars: list[scal.DataScalar] = list(data_scalars)
        self.optimizer: Optimizer = optimizer

    def forward_prop(self, x: np.ndarray):
        if x.shape[1] != self.w.shape[0]:
            raise ModelParameterError(
                f"Shape of x input ({x.shape}) isn't supported by the model. Has to be (m, {self.w.shape[0]})"
            )
        return self.activation(x @ self.w + self.b)

    def back_prop(self, x: np.ndarray,
                  y: np.ndarray,
                  prediction: np.ndarray,
                  loss_enum: LossEnum,
                  learning_rate: float):
        if loss_enum not in self.loss_functions:
            raise ModelParameterError(
                f"Wrong loss function is passed. Linear regressor supports only these - {self.loss_functions}"
            )
        if y.shape[1] != 1:
            raise ModelParameterError(
                f"Shape of y ({y.shape}) is not supported by the model. Has to be ({x.shape[0]}, 1))"
            )

        loss = loss_enum.value
        loss_value = loss(prediction, y)
        dw, db = loss.gradient_values(x, y, prediction)

        return dw * learning_rate, db * learning_rate, loss_value

    def update_parameters(self, dw: np.ndarray, db: float):
        self.w -= dw
        self.b -= db

    def _set_scale_data(self, data: np.ndarray):
        for i in range(len(self.scalars)):
            self.scalars[i].set_values(data)
            data = self.scalars[i](data)

    def _scale_data(self, data: np.ndarray):
        for scalar in self.scalars:
            data = scalar(data)

        return data

    def _validate(self, x_valid: np.ndarray, y_valid: np.ndarray, loss: LossFunction):
        val_prediction = self.forward_prop(x_valid)
        return loss(val_prediction, y_valid)

    def print_fit_progress(self, epoch: int, loss_name: str):
        print(f"[Epoch {epoch}]", end="\t")
        print(f"[loss ({loss_name}) - {self.history['loss'][epoch - 1]}]\t")
        print(f"[val_loss ({loss_name}) - {self.history['val_loss'][epoch - 1]}]\n")

    def predict(self, x):
        x = self._scale_data(x)
        return self.forward_prop(x)

    def fit(self, x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            validation_part: float = 0.2,
            validation_splitter: ds.DataSplitter = ds.RegularValidation()):
        loss = self.optimizer.get_loss_enum()

        self._set_scale_data(x)
        x = self._scale_data(x)

        self.history["loss"] = [0] * epochs
        self.history["val_loss"] = [0] * epochs

        for x_train, x_valid, y_train, y_valid, epoch in validation_splitter(x, y, validation_part, epochs):
            train_loss_value = self.optimizer.optimize(x_train, y_train, self)
            val_loss_value = self._validate(x_valid, y_valid, loss.value)

            self.history["loss"][epoch - 1] = train_loss_value
            self.history["val_loss"][epoch - 1] = val_loss_value

            self.print_fit_progress(epoch, loss.__class__.__name__)

        return self.history


if __name__ == "__main__":
    from random import randint

    x = [[randint(-10, 10), randint(-10, 10), randint(-10, 10)] for i in range(-500, 500)]
    y = [0] * 1000
    for i in range(1000):
        y[i] = 1 if sum(x[i]) > 0 else 0

    x = np.array(x)
    y = np.array(y, ndmin=2).T

    model = BinaryClassifier(3, optimizer=SGD(loss_enum=LossEnum.CROSS_ENTROPY, learning_rate=1e-2),
                             data_scalars=(scal.Standardizer(),))
    history = model.fit(x, y, epochs=100)

    plot_loss_history(history)

    test_x = np.array([[-1, -2, 1],
                       [2, 2, 2],
                       [3, 5, 3],
                       [4, 0, -4],
                       [5, 3, 5]])

    print(model.predict(test_x))

    print("params")
    print(model.w)
    print(model.b)
