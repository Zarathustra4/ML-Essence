import numpy as np

from models.loss_functions import LossFunctionsEnum


class SGD:
    def __init__(self, loss: LossFunctionsEnum = LossFunctionsEnum.MEAN_SQUARED_ERROR,
                 learning_rate: float = 1e-4,
                 batch_size: int = 200):
        self.loss = loss.value
        self.lr = learning_rate
        self.batch_size = batch_size

    def get_grad_values(self, x: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> tuple:
        train_size = x.shape[0]
        stochastic_dw = 0
        stochastic_db = 0
        count = 1

        for i in range(0, train_size, self.batch_size):
            x_batch = x[i: i + self.batch_size]
            y_batch = y[i: i + self.batch_size]
            pred_batch = prediction[i: i + self.batch_size]

            dw, db = self.loss.gradient_values(x_batch, y_batch, pred_batch)
            stochastic_dw += dw * self.lr
            stochastic_db += db * self.lr
            count += 1

        return stochastic_dw / count, stochastic_db / count

    def set_learning_rate(self, learning_rate: float):
        self.lr = learning_rate
