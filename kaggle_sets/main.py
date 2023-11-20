from data_preparation.numpy_cast import DatasetToNumpy
from models.simple_lin_regressor import SimpleLinRegressor
from plot.graph_plot import plot_loss_history
import models.datasplits as ds
import models.data_scalar as scal

if __name__ == "__main__":
    dtnp = DatasetToNumpy("mosquito-indicator", ',')
    x, y = dtnp(["date"], "mosquito_Indicator")


    model = SimpleLinRegressor(units=4)
    history = model.fit(x, y, epochs=100, learning_rate=3e-3, validation_part=0.2,
                        validation_type=ds.ValDataSplitEnum.REGULAR_VAL,
                        scalars=(scal.Standardizer(), scal.Normalizer()))
    plot_loss_history(history)

