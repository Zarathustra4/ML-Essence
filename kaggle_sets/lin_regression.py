from kaggle_sets.model_service.regression_service import RegressionService


def train_save_regressor():
    service = RegressionService()

    history = service.create_train_model()
    metrics = service.test_model()

    print(f"| Prediction Mean Squared Error | {metrics['mae'][0]: .2f}")
    print(f"| Prediction R Squared          | {metrics['r2']: .2f}")
    print(f"| Prediction Mean Squared Error | {metrics['mse']: .2f}")
    print(f"| Final Mean Absolute Error     | {history['mean_absolute_error'][-1][0]: .2f}")
