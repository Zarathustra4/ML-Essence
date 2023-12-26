from time_series.model import train_save_model, validate_model
from time_series.data_preparation import get_data


def train_save_forecaster():
    train_set, validation_set = get_data()
    train_save_model(train_set, epochs=50)
    validate_model(validation_set)
