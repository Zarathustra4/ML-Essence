import os

# DATASETS
BASE_DATASET_PATH = (os.environ.get("BASE_DATASET_PATH") or
                     r"E:\Лабораторні\3 курс\ML Essencials\kaggle_sets\datasets")

# CUSTOM MODELS
LIN_REGRESSOR_PATH = (os.environ.get("LIN_REGRESSOR_PATH") or
                      r"E:\Лабораторні\3 курс\ML Essencials\kaggle_sets\trained_models\regressor.json")
BIN_CLASSIFIER_PATH = (os.environ.get("BIN_CLASSIFIER_PATH") or
                       r"E:\Лабораторні\3 курс\ML Essencials\kaggle_sets\trained_models\classifier.json")

# TIME SERIES
TS_WINDOW_SIZE = 128
TS_BATCH_SIZE = 32
TS_SHUFFLE_BUFFER_SIZE = 1000
TS_MODEL_PATH = (os.environ.get("TS_MODEL_PATH") or
                 r"E:\Лабораторні\3 курс\ML Essencials\kaggle_sets\trained_models\time_series.h5")
TS_DATASET_PATH = r"E:\Лабораторні\3 курс\ML Essencials\kaggle_sets\datasets\daily-min-temperatures.csv"
TS_SPLIT_TIME = 2500
