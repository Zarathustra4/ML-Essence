from enum import Enum

from model_service.classifier_service import train_save_classifier
from model_service.clusterer_service import train_save_clusterer
from model_service.forecast_service import train_save_forecaster
from model_service.regression_service import train_save_regressor


# TODO: rename "kaggle_sets" module

class Mode(Enum):
    LINEAR_REGRESSION = "linear regression"
    BINARY_CLASSIFICATION = "binary classification"
    TIME_SERIES = "time series"
    CLUSTERING = "clustering"


def main(mode: Mode):
    if mode == Mode.LINEAR_REGRESSION:
        train_save_regressor()
    elif mode == Mode.BINARY_CLASSIFICATION:
        train_save_classifier()
    elif mode == Mode.TIME_SERIES:
        train_save_forecaster()
    elif mode == Mode.CLUSTERING:
        train_save_clusterer()


if __name__ == "__main__":
    main(mode=Mode.BINARY_CLASSIFICATION)
