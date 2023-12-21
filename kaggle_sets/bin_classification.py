from data_preparation.dataset_to_numpy import DatasetToNumpy
from processing.models.binaryclassifier import BinaryClassifier
from processing.functions.loss_functions import LossEnum
from processing.models.optimizers import SGD
from plot.graph_plot import plot_loss_history
import processing.preprocessing.data_scalar as scal
from processing.functions.metrics import Accuracy

if __name__ == "__main__":
    dtnp = DatasetToNumpy("water-quality", csv_delimeter=",")
    (x_train, y_train), (x_test, y_test) = dtnp(drop_list=[], y_column="is_safe")

    model = BinaryClassifier(
        units=20,
        optimizer=SGD(loss_enum=LossEnum.CROSS_ENTROPY,
                      learning_rate=1e-2,
                      batch_size=128),
        data_scalars=(scal.Standardizer(),)
    )

    history = model.fit(x_train, y_train, epochs=300)

    plot_loss_history(history)

    model.save("bin.json")

    model = BinaryClassifier(units=20)

    model.load("bin.json")

    test_prediction = model.predict(x_test)

    acc = Accuracy()(test_prediction, y_test)
    print(f"[ACCURACY] - {acc * 100:.2f}%")
