from data_preparation.numpy_cast import DatasetToNumpy
from models.simple_lin_regressor import SimpleLinRegressor
from plot.graph_plot import plot_loss_history
import models.datasplits as ds
import numpy as np
import random


if __name__ == "__main__":
    x, y = cast_dataset_to_numpy() # Change this =)

    y = np.expand_dims(y, axis=1)

    model = SimpleLinRegressor(units=11)
    history = model.fit(x, y, epochs=500, learning_rate=1e-8, validation_part=0.2,
                        validation_type=ds.ValDataSplitEnum.CROSS_VAL)
    plot_loss_history(history)

    test_idx = random.randint(0, x.shape[0])
    x_test = np.array(x[test_idx: test_idx + 21], ndmin=2)
    print(test_idx)
    predictions = model.predict(x_test)
    for i in range(21):
        print(y[test_idx + i][0], round(predictions[i, 0]))

    print(model.w)
