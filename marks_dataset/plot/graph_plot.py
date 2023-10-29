import matplotlib.pyplot as plt
from models.loss_functions import LossFunctions


def plot_history(history, key=LossFunctions.MEAN_SQUARED_ERROR):
    time = [i for i in range(len(history[key]))]
    plt.plot(time, history[key])
    plt.show()
