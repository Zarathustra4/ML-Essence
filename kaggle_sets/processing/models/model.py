from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    @abstractmethod
    def forward_prop(self, x):
        """
        Returns predicted data
        :param x: numpy.ndarray - training data
        :return: numpy.ndarray - predicted data
        """
        ...

    @abstractmethod
    def back_prop(self, x, y, prediction, loss_enum, learning_rate):
        """
        Optimizes model's parameters
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param prediction: np.ndarray - predicted value
        :param loss_enum: LossEnum - enum of loss functions
        :param learning_rate: float
        :return: (dw, db, loss_value)
        """
        ...

    @abstractmethod
    def fit(self, x, y, epochs, validation_part, validation_splitter):
        """
        Fits model to the train data
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param epochs: number of epochs
        :param validation_part: float
        :param validation_splitter: DataSplitter - object, which split data onto train and validation
        :return: dict - history of training
        """
        ...

    @abstractmethod
    def predict(self, x):
        """
        Predicts the value
        :param x: numpy.ndarray - training data
        :return: numpy.ndarray - predicted data
        """
        ...

    @abstractmethod
    def update_parameters(self, dw: np.ndarray, db: float):
        """
        Optimizes the parameters of model
        :param dw: derivative by weights
        :param db: derivative by bias
        :return:
        """

    @abstractmethod
    def save(self, path: str):
        """
        Save model to a given location
        :param path: file location
        :return:
        """

    @abstractmethod
    def load(self, path: str):
        """
        Loads model from a given file location
        :return: None
        """