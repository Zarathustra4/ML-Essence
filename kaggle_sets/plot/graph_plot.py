import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(history, loss_key="loss", val_loss_key="val_loss"):
    time = range(len(history[loss_key]))
    plt.plot(time, history[loss_key], 'b', label="Training loss")
    if history.get(val_loss_key):
        plt.plot(time, history[val_loss_key], 'r', label="Validation loss")
    plt.title("Loss history")
    plt.legend()
    plt.show()


def plot_metric_history(history: dict, metric_name: str = "mse"):
    time = range(len(history[metric_name]))
    plt.plot(time, history[metric_name], 'b', label=f"Training {metric_name}")
    if history.get(f"val_{metric_name}"):
        plt.plot(time, history[f"val_{metric_name}"], 'r', label=f"Validation {metric_name}")
    plt.title(f"{metric_name.capitalize()} history")
    plt.legend()
    plt.show()


def plot_time_series(x, y):
    plt.plot(x, y)
    plt.show()
