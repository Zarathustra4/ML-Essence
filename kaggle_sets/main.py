from data_preparation.dataset_to_numpy import DatasetToNumpy
from models.loss_functions import MSE
from models.optimizers import SGD
from models.simple_lin_regressor import SimpleLinRegressor
from plot.graph_plot import plot_loss_history
import models.data_scalar as scal

if __name__ == "__main__":
    dtnp = DatasetToNumpy("winequality-red", csv_delimeter=";")
    (x_train, y_train), (x_test, y_test) = dtnp(["pH", "free sulfur dioxide", "chlorides"], y_column="quality")

    model = SimpleLinRegressor(units=8,
                               optimizer=SGD(batch_size=100, learning_rate=1e-3))

    history = model.fit(x_train, y_train, epochs=500, scalars=(scal.Normalizer(),))

    plot_loss_history(history)

    test_prediction = model.predict(x_test)
    test_loss = MSE()(test_prediction, y_test)

    print(f"[TEST LOSS] - {test_loss}")
