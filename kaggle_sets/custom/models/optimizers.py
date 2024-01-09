from abc import ABC, abstractmethod

import numpy as np

from kaggle_sets.custom.functions.loss_functions import LossEnum
from kaggle_sets.custom.models.model import Model


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, x: np.ndarray, y: np.ndarray, model: Model):
        ...

    @abstractmethod
    def get_learning_rate(self) -> float:
        ...

    @abstractmethod
    def get_loss_enum(self) -> LossEnum:
        ...


class GradientDescent(Optimizer):
    def __init__(self, loss_enum: LossEnum = LossEnum.MEAN_SQUARED_ERROR,
                 learning_rate: float = 1e-8):
        self.loss_enum = loss_enum
        self.lr = learning_rate

    def optimize(self, x: np.ndarray, y: np.ndarray, model: Model):
        prediction = model.forward_prop(x)
        dw, db, train_loss_value = model.back_prop(x, y, prediction, self.loss_enum, self.lr)
        model.update_parameters(dw, db)
        return train_loss_value

    def get_learning_rate(self) -> float:
        return self.lr

    def get_loss_enum(self) -> LossEnum:
        return self.loss_enum


class SGD(Optimizer):
    def __init__(self, loss_enum: LossEnum,
                 learning_rate: float = 1e-4,
                 batch_size: int = 200):
        self.loss_enum: LossEnum = loss_enum
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
            dw, db, _ = model.back_prop(x_batch, y_batch, prediction, self.loss_enum, self.lr)
            model.update_parameters(dw, db)

        return self.loss_enum.value(model.forward_prop(x), y)

    def get_learning_rate(self) -> float:
        return self.lr

    def get_loss_enum(self) -> LossEnum:
        return self.loss_enum
