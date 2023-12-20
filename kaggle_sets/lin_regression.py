from data_preparation.dataset_to_numpy import DatasetToNumpy
from processing.functions.loss_functions import LossEnum
from processing.models.optimizers import SGD
from processing.models.simple_lin_regressor import SimpleLinRegressor
from plot.graph_plot import plot_loss_history
from processing.functions.metrics import MetricsEnum
import processing.preprocessing.data_scalar as scal

if __name__ == "__main__":
    dtnp = DatasetToNumpy("mosquito-indicator", csv_delimeter=",")
    (x_train, y_train), (x_test, y_test) = dtnp(["date"], y_column="mosquito_Indicator")

    model = SimpleLinRegressor(
        units=4,
        data_scalars=(scal.Normalizer(),),
        optimizer=SGD(
            loss_enum=LossEnum.MEAN_SQUARED_ERROR,
            batch_size=128,
            learning_rate=1e-3
    ))

    history = model.fit(x_train, y_train, epochs=300, metrics=(MetricsEnum.MEAN_SQUARED_ERROR.value, MetricsEnum.MEAN_ABSOLUTE_ERROR.value))

    plot_loss_history(history)

    test_prediction = model.predict(x_test)

    print(f"| Prediction Mean Squared Error | {MetricsEnum.MEAN_SQUARED_ERROR.value(test_prediction, y_test)}")
    print(f"| Prediction R Squared          | {MetricsEnum.R_SQUARED.value(test_prediction, y_test)}")
    print(f"| Final Mean Squared Error      | {history['mean_squared_error'][-1]}")
    print(f"| Final Mean Absolute Error     | {history['mean_absolute_error'][-1]}")
