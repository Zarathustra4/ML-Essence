from enum import Enum

from model_service.classifier_service import train_save_classifier
from model_service.clusterer_service import train_save_clusterer
from model_service.forecast_service import train_save_forecaster
from model_service.regression_service import train_save_regressor


# TODO: rename "kaggle_sets" module


def train_all_models():
    train_save_regressor()
    train_save_classifier()
    train_save_forecaster()
    train_save_clusterer()


class Mode(Enum):
    LINEAR_REGRESSION = train_save_regressor
    BINARY_CLASSIFICATION = train_save_classifier
    TIME_SERIES = train_save_forecaster
    CLUSTERING = train_save_clusterer
    TRAIN_ALL = train_all_models


def main(*modes):
    for mode in modes:
        mode()


if __name__ == "__main__":
    main(Mode.LINEAR_REGRESSION, Mode.BINARY_CLASSIFICATION)
