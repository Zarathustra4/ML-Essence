import pandas as pd
from pathlib import Path
import numpy as np


"""
Usage:
    dtnp = DatasetToNumpy("mosquito-indicator", ',')
    x, y = dtnp(["date"], "mosquito_Indicator")
"""


class DatasetToNumpy:
    def __init__(self, csv_file: str, csv_delimeter: str) -> None:
        self.csv = Path(f"./datasets/{csv_file}.csv")
        self.csv_delimeter = csv_delimeter

    def __call__(self, drop_list: list, y_column: str, test_size: float = 0.2, random_seed: int = 42) -> tuple:
        return self._cast(drop_list, y_column, test_size, random_seed)

    def _cast(self, drop_list: list, y_column: str, test_size: float, random_seed: int) -> tuple:
        dataframe = pd.read_csv(self.csv, delimiter=self.csv_delimeter)

        drop_list.append(y_column)
        x = dataframe.drop(drop_list, axis=1)
        y = dataframe[y_column]

        np_x = x.to_numpy()
        np_y = y.to_numpy()
        np_y = np.expand_dims(y, axis=1)

        train_x, test_x, train_y, test_y = self._split_data(np_x, np_y, test_size, random_seed)
        np.set_printoptions(suppress=True, formatter={'float_kind': '{:.1f}'.format})
        return (train_x, train_y), (test_x, test_y)

    def _split_data(self, np_x, np_y, test_size, random_seed):
        np.random.seed(random_seed)
        indices = np.random.permutation(len(np_x))
        test_size = int(len(np_x) * test_size)

        train_indices, test_indices = indices[test_size:], indices[:test_size]
        train_x, test_x = np_x[train_indices], np_x[test_indices]
        train_y, test_y = np_y[train_indices], np_y[test_indices]

        return train_x, test_x, train_y, test_y





