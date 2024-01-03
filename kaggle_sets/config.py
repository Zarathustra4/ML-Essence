import os

BASE_PATH = os.path.abspath(os.getcwd())

# DATASETS
BASE_DATASET_PATH = os.path.join(BASE_PATH, "datasets")

# CUSTOM MODELS
LIN_REGRESSOR_PATH = os.path.join(BASE_PATH, "trained_models", "regressor.json")
BIN_CLASSIFIER_PATH = os.path.join(BASE_PATH, "trained_models", "classifier.json")

# TIME SERIES
TS_WINDOW_SIZE = 128
TS_BATCH_SIZE = 32
TS_SHUFFLE_BUFFER_SIZE = 1000
TS_SPLIT_TIME = 2500

TS_MODEL_PATH = os.path.join(BASE_PATH, "trained_models", "time_series.h5")
TS_DATASET_PATH = os.path.join(BASE_PATH, "datasets", "daily-min-temperatures.csv")
