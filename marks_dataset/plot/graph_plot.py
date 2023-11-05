import matplotlib.pyplot as plt


def plot_loss_history(history, loss_key="loss", val_loss_key="val_loss"):
    time = [i for i in range(len(history[loss_key]))]
    plt.plot(time, history[loss_key], 'b')
    plt.plot(time, history[val_loss_key], 'r')
    plt.show()
