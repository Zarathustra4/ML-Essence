import numpy as np

from models.loss_functions import fix_dims, LossFunctions, LossFunction
from exceptions.exceptions import ModelParameterError
from model import Model
from plot import graph_plot


class SimpleLinRegressor(Model):
    def __init__(self, units=2):
        self.w = np.random.rand(units, 1)
        self.b = 0
        self.loss_functions = (LossFunctions.MEAN_SQUARED_ERROR,)  # tuple of allowed loss functions
        self.history = {}

    def forward_prop(self, x):
        x = fix_dims(x)
        return self.w.T @ x + self.b

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

        weight_lr = learning_rate
        bias_lr = learning_rate * 1e4  # bias requires larger learning rate

        dw, db = loss_function.gradient_values(x, y, prediction)

        return dw * weight_lr, db * bias_lr, loss_value

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, loss: LossFunctions, learning_rate=0.001):
        self.history[loss] = [0] * epochs

        for epoch in range(1, epochs):
            prediction = self.forward_prop(x)
            dw, db, loss_value = self.back_prop(x, y, prediction, loss, learning_rate)
            self.w -= dw.T
            self.b -= db
            self.history[loss][epoch - 1] = loss_value
            print(f"[Epoch {epoch}]", end="\t")
            print(f"[Weight]-{self.w} | [Bias]-{self.b}")
            print(f"[Loss ({loss.name})-{loss_value}]")

        return self.history

    def predict(self, x: np.ndarray):
        return self.forward_prop(x)


if __name__ == "__main__":

    x_test = [[i, i / 2, i / 3] for i in range(1000)]
    y_test = [num[0] + 2 * num[1] - 1 for num in x_test]
    x_test = np.array(x_test).T
    y_test = np.array(y_test)

    model = SimpleLinRegressor(units=3)

    history = model.fit(x_test, y_test, epochs=150, loss=LossFunctions.MEAN_SQUARED_ERROR, learning_rate=1.5e-6)
    graph_plot.plot_history(history, LossFunctions.MEAN_SQUARED_ERROR)

