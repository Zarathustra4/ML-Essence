import pandas as pd
import numpy as np
from pathlib import Path


def cast_dataset_to_numpy(debug: bool = False) -> tuple:
    """
    Returns tuple of numpy arrays, x; y accordingly.
    """
    file = Path("./datasets/winequality-red.csv")

    dataframe = pd.read_csv(file, delimiter=';')
    x = dataframe.drop("quality", axis=1) # <- Truncate 'quality' column
    y = dataframe['quality'] # <- Move 'quality' to y instead
    
    """ Convert dataframes to numpy arrays """
    x_np_array = x.to_numpy()
    y_np_array = y.to_numpy()

    if debug:
        """ Debug dataframe separation """
        print(f"<Data> X (features):\n{x.head()}\n")
        print(f"<Quality> Y (target):\n{y.head()}\n")

        """ Debug numpy cast """
        print(f"<Data> X Numpy Array (features):\n{x_np_array[:5, :]}\n")
        print(f"<Quality> Y Numpy Array (target):\n{y_np_array[:5]}\n")
    
    return x_np_array, y_np_array
