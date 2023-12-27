import numpy as np

from kaggle_sets.data_preparation.dataset_to_numpy import DatasetToNumpy
import kaggle_sets.config as conf
from kaggle_sets.plot.graph_plot import plot_loss_history
from kaggle_sets.processing.functions.loss_functions import LossEnum
from kaggle_sets.processing.functions.metrics import Accuracy
from kaggle_sets.processing.models.binaryclassifier import BinaryClassifier
from kaggle_sets.processing.models.optimizers import SGD
import kaggle_sets.processing.preprocessing.data_scalar as scal


class ClassifierService:
    def __init__(self):
        self.numpy_caster = DatasetToNumpy("water-quality", csv_delimeter=",")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.numpy_caster(drop_list=[], y_column="is_safe")

        self.path = conf.BIN_CLASSIFIER_PATH

    def _train_model(
            self,
            model: BinaryClassifier,
            epochs: int,
            validation_split: float,
            plot_history: bool
    ):
        history = model.fit(self.x_train, self.y_train, epochs, validation_split)

        if plot_history:
            plot_loss_history(history)

        model.save(self.path)

        return history

    def _get_model(self, load=True):
        model = BinaryClassifier(
            units=20,
            optimizer=SGD(loss_enum=LossEnum.CROSS_ENTROPY,
                          learning_rate=1e-2,
                          batch_size=128),
            data_scalars=(scal.Standardizer(),)
        )

        if load:
            model.load(self.path)

        return model

    def create_empty_model(self) -> None:
        """
            Creates an untrained model
            :return: None
        """
        model = self._get_model(load=False)
        model.save(self.path)

    def train_model(
            self,
            epochs: int = 300,
            validation_split: float = 0.2,
            plot_history=True
    ) -> dict:
        """
        Trains a model
        :param epochs: number of epochs
        :param validation_split: validation part of dataset
        :param plot_history: set True if you want to plot training history
        :return: dict - training history
        """
        model = self._get_model()

        return self._train_model(
            model=model,
            epochs=epochs,
            validation_split=validation_split,
            plot_history=plot_history
        )

    def create_train_model(
            self,
            epochs: int = 300,
            validation_split: float = 0.2,
            plot_history=True
    ) -> dict:
        """
            Creates and trains a model
            :param epochs: number of learning repetitions
            :param validation_split: float value, percent of validating data
            :param plot_history: True if user wants to plot train history, else - False
            :return: history - dict
        """

        model = self._get_model(load=False)

        return self._train_model(
            model=model,
            epochs=epochs,
            validation_split=validation_split,
            plot_history=plot_history
        )

    def test_model(self) -> dict:
        model = self._get_model()

        predictions = model.predict(self.x_test)

        acc = Accuracy()

        return {
            "accuracy": acc(predictions, self.y_test)
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        model = self._get_model()

        return model.predict(x)


if __name__ == "__main__":
    service = ClassifierService()

    print(
        service.create_empty_model()
    )

    print(service.train_model())

    service.create_train_model()

    service.test_model()

    print(
        service.predict(
            service.x_train[:5]
        )
    )
