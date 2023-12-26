import pandas as pd
import tensorflow as tf
import kaggle_sets.config as conf


def parse_data(filename=conf.TS_DATASET_PATH):
    temps = pd.read_csv(filename, delimiter=",")["Temp"].to_numpy()
    return temps


def window_data(series,
                window_size=conf.TS_WINDOW_SIZE,
                batch_size=conf.TS_BATCH_SIZE,
                shuffle_buffer=conf.TS_SHUFFLE_BUFFER_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


def split_data(series, split_time=conf.TS_SPLIT_TIME):
    train_series = series[:split_time]
    validation_series = series[split_time:]
    return train_series, validation_series


def get_data():
    series = parse_data()
    train_series, validation_series = split_data(series)
    return window_data(train_series), validation_series
