import numpy as np
from enum import Enum


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
        train_size = x.shape[0]
        data_parts_number = int(1 / validation_part)
        data_part_size = int(train_size * validation_part)

        data = np.concatenate((x, y), axis=1)
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1:]

        part = 0
        for epoch in range(1, epochs + 1):
            if part > data_parts_number:
                part = 0
            start_val_idx = part * data_part_size
            end_val_idx = start_val_idx + data_part_size
            x_valid = x[start_val_idx: end_val_idx, :]
            y_valid = y[start_val_idx: end_val_idx, :]
            x_train = np.concatenate((x[0: start_val_idx, :], x[end_val_idx:, :]), axis=0)
            y_train = np.concatenate((y[0: start_val_idx, :], y[end_val_idx:, :]), axis=0)

            part += 1
            yield x_train, x_valid, y_train, y_valid, epoch


class ValDataSplitEnum(Enum):
    REGULAR_VAL = RegularValidation()
    CROSS_VAL = CrossValidation()
