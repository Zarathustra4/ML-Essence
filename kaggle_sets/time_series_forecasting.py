from kaggle_sets.model_service.forecast_service import ForecastService


def train_save_forecaster():
    service = ForecastService()
    service.reset_model()
    service.train_model(epochs=40)

    print("| --- Testing model --- |")
    print(service.test_model())
