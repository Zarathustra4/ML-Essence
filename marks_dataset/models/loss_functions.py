import numpy as np


def fix_dims(vec: np.ndarray, axis=0):
    return np.expand_dims(vec, axis=axis) if len(vec.shape) == 1 else vec


class LossFunction:
    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        """
        :return: value of loss function
        """
        ...

    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        return self.loss_function(prediction, y)

    def gradient_values(self):
        """
        :return: value of partial derivatives of loss function by W and b
        """
        ...


class MSE(LossFunction):
    def loss_function(self, prediction: np.ndarray, y: np.ndarray):
        prediction = fix_dims(prediction)
        diff = prediction - y
        m = diff.shape[1]
        return (1 / m) * (diff @ diff.T)[0, 0]


if __name__ == "__main__":
    mse = MSE()
    y = np.array([1, 2, 3])
    pred = np.array([2, 2, 1])
    print(mse.loss_function(y, pred))
