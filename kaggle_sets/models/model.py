import numpy as np


class Model:
    def forward_prop(self, x):
        """
        Returns predicted data
        :param x: numpy.ndarray - training data
        :return: numpy.ndarray - predicted data
        """
        ...

    def back_prop(self, x, y, prediction, loss, learning_rate):
        """
        Optimizes model's parameters
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param prediction: np.ndarray - predicted value
        :param loss: LossFunctionsEnum - enum of loss functions
        :param learning_rate: float
        :return: (dw, db, loss_value)
        """
        ...

    def fit(self, x, y, epochs, loss, learning_rate):
        """
        Fits model to the train data
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param epochs: number of epochs
        :param loss: LossFunctionsEnum - enum of loss functions
        :param learning_rate: float
        :return: dict - history of training
        """
        ...

    def predict(self, x):
        """
        Predicts the value
        :param x: numpy.ndarray - training data
        :return: numpy.ndarray - predicted data
        """
        ...

    def _split_data(self, x: np.ndarray, y: np.ndarray, validation_part: float) -> tuple:
        """
        Splits the data into train and validation sets
        :param x: np.ndarray - train input data
        :param y: np.ndarray - train output data
        :param validation_part: float
        :return: (x_train, x_test, y_train, y_test)
        """
        ...


