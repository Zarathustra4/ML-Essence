import numpy as np
import pandas as pd

from kaggle_sets.data_preparation.dataset_to_numpy import DatasetToNumpy
import kaggle_sets.config as conf
from kaggle_sets.plot.graph_plot import plot_loss_history, plot_metric_history
from kaggle_sets.custom.functions.loss_functions import LossEnum
from kaggle_sets.custom.functions.metrics import Accuracy, ConfusionMatrix, precision, recall, f1, MetricsEnum
from kaggle_sets.custom.models.binaryclassifier import BinaryClassifier
from kaggle_sets.custom.models.optimizers import SGD
import kaggle_sets.custom.preprocessing.data_scalar as scal
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class ClassifierService:
    def __init__(self):
        self.numpy_caster = DatasetToNumpy("water-quality", csv_delimeter=",")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.numpy_caster(drop_list=[], y_column="is_safe")

        self.path = conf.BIN_CLASSIFIER_PATH
        self.model: BinaryClassifier = self._get_model()
        self.confusion_matrix = ConfusionMatrix()

    @staticmethod
    def get_untrained_model():
        return BinaryClassifier(
            units=20,
            optimizer=SGD(loss_enum=LossEnum.CROSS_ENTROPY,
                          learning_rate=1e-2,
                          batch_size=128),
            data_scalars=(scal.Standardizer(),)
        )

    def train_model(
            self,
            epochs: int = 300,
            validation_split: float = 0.2,
            plot_history=True,
            metrics=(MetricsEnum.ACCURACY.value,)
    ):
        """
        Trains a model
        :param epochs: number of epochs
        :param validation_split: validation part of dataset
        :param plot_history: set True if you want to plot training history
        :param metrics: tuple of metrics
        :return: dict - training history
        """
        history = self.model.fit(self.x_train, self.y_train, epochs, validation_split, metrics=metrics)

        if plot_history:
            plot_loss_history(history)
            plot_metric_history(history, "accuracy")

        self.model.save(self.path)

        return history

    def _get_model(self):
        model = ClassifierService.get_untrained_model()

        if Path(self.path).exists():
            model.load(self.path)

        return model

    def reset_model(self) -> None:
        """
            Creates an untrained model
            :return: None
        """
        self.model = ClassifierService.get_untrained_model()

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

        self.model = ClassifierService.get_untrained_model()

        return self.train_model(
            epochs=epochs,
            validation_split=validation_split,
            plot_history=plot_history
        )

    def _roc(self, predictions, plot=True):
        fpr, tpr, thresholds = roc_curve(self.y_test, predictions)

        roc_auc = auc(fpr, tpr)

        if plot:
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()

        return roc_auc

    def test_model(self) -> dict:
        """
        Tests model on prepared test set and calculates metrics
        :return: dict of metrics
        """
        predictions = self.model.predict(self.x_test)

        acc = Accuracy()

        confusion = self.confusion_matrix(predictions, self.y_test)

        roc_auc = self._roc(predictions)

        return {
            "accuracy": acc(predictions, self.y_test),
            "confusion": confusion,
            "precision": precision(predictions, self.y_test),
            "recall": recall(predictions, self.y_test),
            "f1": f1(predictions, self.y_test),
            "auc": roc_auc
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions
        :param x: input data
        :return: prediction
        """

        return self.model.predict(x)

    def predict_by_csv(self, path: str) -> np.ndarray:
        df = pd.read_csv(path)
        x = df.to_numpy()

        return self.model.predict(x).round()


def train_save_classifier():
    service = ClassifierService()
    history = service.create_train_model(epochs=1700)
    metrics = service.test_model()

    print(f"| Prediction Accuracy       | {metrics['accuracy']: .3f}")
    print(f"| Final Loss                | {history['loss'][-1][0]: .3f}")
    print(f"| Precision                 | {metrics['precision']: .3f}")
    print(f"| Recall                    | {metrics['recall']: .3f}")
    print(f"| F1                        | {metrics['f1']: .3f}")
    print(f"| Area under the roc curve  | {metrics['auc']: .3f}")

    ConfusionMatrix.print_matrix(metrics['confusion'])
