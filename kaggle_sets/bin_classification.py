from kaggle_sets.model_service.classifier_service import ClassifierService
from kaggle_sets.processing.functions.metrics import ConfusionMatrix


def train_save_classifier():
    service = ClassifierService()
    history = service.create_train_model()
    metrics = service.test_model()

    print(f"| Prediction Accuracy   | {metrics['accuracy']: .2f}")
    print(f"| Final Loss            | {history['loss'][-1][0]: .2f}")
    ConfusionMatrix.print_matrix(metrics['confusion'])
