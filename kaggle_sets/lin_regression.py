from data_preparation.dataset_to_numpy import DatasetToNumpy
from processing.functions.loss_functions import MSE, LossEnum
from processing.functions.metrics import R2
from processing.models.optimizers import SGD
from processing.models.simple_lin_regressor import SimpleLinRegressor
from plot.graph_plot import plot_loss_history
import processing.preprocessing.data_scalar as scal

if __name__ == "__main__":
    dtnp = DatasetToNumpy("mosquito-indicator", csv_delimeter=",")
    (x_train, y_train), (x_test, y_test) = dtnp(["date"], y_column="mosquito_Indicator")

    model = SimpleLinRegressor(units=4,
                               data_scalars=(scal.Normalizer(),),
                               optimizer=SGD(loss_enum=LossEnum.MEAN_SQUARED_ERROR,
                                             batch_size=128, learning_rate=1e-3))

    history = model.fit(x_train, y_train, epochs=300)

    plot_loss_history(history)

    test_prediction = model.predict(x_test)
    r2 = R2()
    mse = MSE()

    print(f"[TEST MSE LOSS] - {mse(test_prediction, y_test)}")
    print(f"[TEST R2 METRIC] - {r2(test_prediction, y_test)}")
