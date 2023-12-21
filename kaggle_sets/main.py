from enum import Enum
from lin_regression import train_save_regressor
from bin_classification import train_save_classifier


class Mode(Enum):
    LINEAR_REGRESSION = "linear regression"
    BINARY_CLASSIFICATION = "binary classification"


def main(mode: Mode):
    if mode == Mode.LINEAR_REGRESSION:
        train_save_regressor()
    elif mode == Mode.BINARY_CLASSIFICATION:
        train_save_classifier()


if __name__ == "__main__":
    main(mode=Mode.LINEAR_REGRESSION)

