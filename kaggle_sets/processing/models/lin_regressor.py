import numpy as np

from kaggle_sets.processing.functions.loss_functions import LossEnum, LossFunction
from kaggle_sets.exceptions.exceptions import ModelParameterError
from kaggle_sets.processing.models.model import Model
import kaggle_sets.processing.preprocessing.datasplits as ds
from kaggle_sets.processing.models.optimizers import SGD, Optimizer
import kaggle_sets.processing.preprocessing.data_scalar as scal
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

        self.history["loss"] = [0] * epochs
        self.history["val_loss"] = [0] * epochs

        metric_names = [metric.name for metric in metrics]
        self.history.update({f"{metric_name}": [0] * epochs for metric_name in metric_names})
        self.history.update({f"val_{metric_name}": [0] * epochs for metric_name in metric_names})

        for x_train, x_valid, y_train, y_valid, epoch in validation_splitter(x, y, validation_part, epochs):
            train_loss_value = self.optimizer.optimize(x_train, y_train, self)
            val_loss_value = self._validate(x_valid, y_valid, loss.value)

            self.history["loss"][epoch - 1] = train_loss_value
            self.history["val_loss"][epoch - 1] = val_loss_value

            for metric in metrics:
                train_metric_value = metric(y_train, self.predict(x_train))
                val_metric_value = metric(y_valid, self.predict(x_valid))
                metric_name = metric.name

                self.history[f"{metric_name}"][epoch - 1] = train_metric_value
                self.history[f"val_{metric_name}"][epoch - 1] = val_metric_value
            
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

    def load(self, path: str):
        with open(path, 'r') as file:
            model_data = json.load(file)
            self.w = np.array(model_data["w"])
            self.b = model_data["b"]

            self.scalars = []
            for json_scalar in model_data["scalars"]:
                self.scalars.append(
                    scal.create_data_scalar(json_scalar)
                )
