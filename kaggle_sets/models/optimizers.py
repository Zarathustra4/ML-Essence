from abc import ABC, abstractmethod

import numpy as np

from models.loss_functions import LossFunctionsEnum
from models.model import Model


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, x: np.ndarray, y: np.ndarray, model: Model):
        ...


class GradientDescent(Optimizer):
    def __init__(self, loss: LossFunctionsEnum = LossFunctionsEnum.MEAN_SQUARED_ERROR,
                 learning_rate: float = 1e-8):
        self.loss = loss
        self.lr = learning_rate

    def optimize(self, x: np.ndarray, y: np.ndarray, model: Model):
        prediction = model.forward_prop(x)
        dw, db, train_loss_value = model.back_prop(x, y, prediction, self.loss, self.lr)
        model.update_parameters(dw, db)
        return train_loss_value


class SGD(Optimizer):
    def __init__(self, loss: LossFunctionsEnum = LossFunctionsEnum.MEAN_SQUARED_ERROR,
                 learning_rate: float = 1e-4,
                 batch_size: int = 200):
        self.loss: LossFunctionsEnum = loss
        self.lr = learning_rate
        self.batch_size = batch_size

    def optimize(self, x: np.ndarray, y: np.ndarray, model: Model):
        train_size = x.shape[0]
        permutation = np.random.permutation(train_size)
        x = x[permutation]
        y = y[permutation]

        for i in range(0, train_size, self.batch_size):
            x_batch = x[i: i + self.batch_size]
            y_batch = y[i: i + self.batch_size]
            prediction = model.forward_prop(x_batch)
            dw, db, _ = model.back_prop(x_batch, y_batch, prediction, self.loss, self.lr)
            model.update_parameters(dw, db)

        return self.loss.value(model.forward_prop(x), y)

