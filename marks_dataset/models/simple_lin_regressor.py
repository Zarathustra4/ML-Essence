import numpy as np


from models.loss_functions import fix_dims, MSE, LossFunctions, LossFunction
from exceptions.exceptions import ModelParameterError
from model import Model


class SimpleLinRegressor(Model):
    def __init__(self, w=1, b=0):
        self.w = w
        self.b = b
        self.loss_functions = (LossFunctions.MEAN_SQUARED_ERROR, )  # tuple of allowed loss functions
        self.history = {}

    def forward_prop(self, x):
        x = fix_dims(x)
        return self.w * x + self.b

    def back_prop(self, x: np.ndarray,
                  y: np.ndarray,
                  prediction: np.ndarray,
                  loss: LossFunctions,
                  learning_rate=0.001):
        if loss not in self.loss_functions:
            raise ModelParameterError(
                f"Wrong loss function is passed. Linear regressor supports only these - {self.loss_functions}"
            )

        x = fix_dims(x)
        y = fix_dims(y)
        prediction = fix_dims(prediction)
        loss_function: LossFunction = loss.value
        loss_value = loss_function(y, prediction)
        dw, db = loss_function.gradient_values(x, y, prediction, self.w, self.b)

        return dw * learning_rate, db * learning_rate, loss_value

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, loss: LossFunctions, learning_rate=0.001):
        self.history[loss] = [0] * epochs

        for epoch in range(1, epochs):
            prediction = self.forward_prop(x)
            dw, db, loss_value = self.back_prop(x, y, prediction, loss, learning_rate)
            self.w -= dw
            self.b -= db
            self.history[loss][epoch - 1] = loss_value
            print(f"[Epoch {epoch}]", end="\t")
            print(f"[Loss ({loss.name}) - {loss_value}]")

        return self.history


if __name__ == "__main__":
    model = SimpleLinRegressor()
    x = [i for i in range(1000)]
    y = [num * 2 - 5 for num in x]
    x = np.array(x)
    y = np.array(y)

    model.fit(x, y, epochs=100, loss=LossFunctions.MEAN_SQUARED_ERROR, learning_rate=1e-6)
