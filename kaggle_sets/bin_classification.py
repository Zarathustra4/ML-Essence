from kaggle_sets.model_service.classifier_service import ClassifierService
from kaggle_sets.processing.functions.metrics import ConfusionMatrix


def train_save_classifier():
    service = ClassifierService()
    history = service.create_train_model()
    metrics = service.test_model()

    print(f"| Prediction Accuracy   | {metrics['accuracy']: .3f}")
    print(f"| Final Loss            | {history['loss'][-1][0]: .3f}")
    print(f"| Precision             | {metrics['precision']: .3f}")
    print(f"| Recall                | {metrics['recall']: .3f}")
    print(f"| F1                    | {metrics['f1']: .3f}")

    ConfusionMatrix.print_matrix(metrics['confusion'])
