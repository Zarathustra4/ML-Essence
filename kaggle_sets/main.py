from enum import Enum
from lin_regression import train_save_regressor
from bin_classification import train_save_classifier
from forecaster import train_save_forecaster


# TODO: rename "kaggle_sets" module

class Mode(Enum):
    LINEAR_REGRESSION = "linear regression"
    BINARY_CLASSIFICATION = "binary classification"
    TIME_SERIES = "time series"


def main(mode: Mode):
    if mode == Mode.LINEAR_REGRESSION:
        train_save_regressor()
    elif mode == Mode.BINARY_CLASSIFICATION:
        train_save_classifier()
    elif mode == Mode.TIME_SERIES:
        train_save_forecaster()


if __name__ == "__main__":
    main(mode=Mode.TIME_SERIES)

