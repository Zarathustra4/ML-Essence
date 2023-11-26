from data_preparation.numpy_cast import DatasetToNumpy
from models.optimizers import SGD
from models.simple_lin_regressor import SimpleLinRegressor
from plot.graph_plot import plot_loss_history
import models.data_scalar as scal

if __name__ == "__main__":
    dtnp = DatasetToNumpy("winequality-red", csv_delimeter=";")
    x, y = dtnp(["pH", "free sulfur dioxide", "chlorides"], y_column="quality")

    model = SimpleLinRegressor(units=8,
                               optimizer=SGD(batch_size=100, learning_rate=1e-2))

    history = model.fit(x, y, epochs=50, scalars=(scal.Normalizer(),))

    plot_loss_history(history)

    count = 0
    prediction = model.predict(x)
    for i in range(len(x)):
        if round(prediction[i, 0]) == y[i, 0]:
            count += 1

    print(f"Accuracy - {count / len(x)}")

    print(f"w -\n {model.w}")
    print(f"b - {model.b}")
