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

    def __call__(self, drop_list: list, y_column: str) -> tuple:
        return self._cast(drop_list, y_column)
        
    def _cast(self, drop_list: list, y_column: str) -> tuple:
        dataframe = pd.read_csv(self.csv, delimiter=self.csv_delimeter)

        drop_list.append(y_column)
        x = dataframe.drop(drop_list, axis=1)
        y = dataframe[y_column]
        
        np_x = x.to_numpy()
        np_y = y.to_numpy()
        np_y = np.expand_dims(y, axis=1)

        return (np_x, np_y)