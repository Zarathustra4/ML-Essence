from data_preparation.numpy_cast import DatasetToNumpy
from models.simple_lin_regressor import SimpleLinRegressor
from plot.graph_plot import plot_loss_history
import models.datasplits as ds
# import models.data_scalar as scal

if __name__ == "__main__":
    dtnp = DatasetToNumpy("mosquito-indicator", ',')
    x, y = dtnp(["date"], "mosquito_Indicator")


    model = SimpleLinRegressor(units=4)
    history = model.fit(x, y, epochs=300, learning_rate=1e-3, validation_part=0.2,
                        validation_type=ds.ValDataSplitEnum.REGULAR_VAL,
                        scalars=(scal.Standardizer(), scal.Normalizer()))
    plot_loss_history(history)

    test_idx = random.randint(0, x.shape[0])
    x_test = np.array(x[test_idx: test_idx + 21], ndmin=2)
    print(test_idx)
    predictions = model.predict(x_test)
    for i in range(21):
        print(y[test_idx + i][0], round(predictions[i, 0]))

    print(model.w)
