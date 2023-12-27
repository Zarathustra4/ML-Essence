import tensorflow as tf
import matplotlib.pyplot as plt
import kaggle_sets.config as conf


def get_model(window_size: int = conf.TS_WINDOW_SIZE):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[window_size, 1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(lr=5e-3),
                  metrics=["mae"])

    return model


def model_forecast(model, series, window_size=conf.TS_WINDOW_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    return model.predict(ds)


def train_save_model(train_set, epochs):
    model = get_model()

    history = model.fit(train_set, epochs=epochs)

    model.save(conf.TS_MODEL_PATH)

    plt.plot(history.history["loss"])
    plt.show()


def get_trained_model():
    return tf.keras.models.load_model(conf.TS_MODEL_PATH)


def validate_model(validation_set):
    model = get_trained_model()
    forecast = model_forecast(model, validation_set)

    validation_set = validation_set[conf.TS_WINDOW_SIZE:]

    plt.figure(figsize=(10, 6))
    plt.plot(validation_set, 'r')
    plt.plot(forecast, 'b')
    plt.show()
