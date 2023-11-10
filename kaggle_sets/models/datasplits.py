import numpy as np


class DataSplitter:
    def _train_val_set(self, x: np.ndarray, y: np.ndarray, validation_part: float, epochs: int):
        ...

    def __call__(self, x: np.ndarray, y: np.ndarray, validation_part: float, epochs: int):
        return self._train_val_set(x, y, validation_part, epochs)


class RegularValidation(DataSplitter):
    @staticmethod
    def _split_data(x: np.ndarray, y: np.ndarray, validation_part: float):
        """
            Splits the data into train and validation sets
            :param x: np.ndarray - train input data
            :param y: np.ndarray - train output data
            :param validation_part: float
            :return: (x_train, x_test, y_train, y_test)
        """
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

    def _train_val_set(self, x: np.ndarray, y: np.ndarray, validation_part: float, epochs: int):
        x_train, x_valid, y_train, y_valid = RegularValidation._split_data(x, y, validation_part)

        for epoch in range(1, epochs + 1):
            yield x_train, x_valid, y_train, y_valid, epoch

class CrossValidation(DataSplitter):
    def _train_val_set(self, x: np.ndarray, y: np.ndarray, validation_part: float, epochs: int):
        ...