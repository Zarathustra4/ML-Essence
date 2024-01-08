import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# DATASETS
BASE_DATASET_PATH = os.path.join(BASE_PATH, "datasets")

# CUSTOM MODELS
LIN_REGRESSOR_PATH = os.path.join(BASE_PATH, "trained_models", "regressor.json")
BIN_CLASSIFIER_PATH = os.path.join(BASE_PATH, "trained_models", "classifier.json")

# CLUSTERING
CLUSTER_DATASET_PATH = os.path.join(BASE_PATH, "datasets", "airline-satisfaction.csv")
CLUSTER_MODEL_PATH = os.path.join(BASE_PATH, "trained_models", "clustering.h5")

# TIME SERIES
TS_WINDOW_SIZE = 150
TS_BATCH_SIZE = 32
TS_SHUFFLE_BUFFER_SIZE = 1000
TS_SPLIT_TIME = 2500

TS_MODEL_PATH = os.path.join(BASE_PATH, "trained_models", "time_series.h5")
TS_DATASET_PATH = os.path.join(BASE_PATH, "datasets", "daily-min-temperatures.csv")
