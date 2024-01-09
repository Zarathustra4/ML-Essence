import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import kaggle_sets.config as conf

def parse_data(filename=conf.CLUSTER_DATASET_PATH):
    data = pd.read_csv(filename, delimiter=",")
    return data


def drop_columns(data):
    return data.drop(['Unnamed: 0', 'id', 'satisfaction'], axis=1)


def encode_categorical(data):
    object_cols = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in object_cols:
        data[col] = label_encoder.fit_transform(data[col])
    return data


def scale_data(data):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return scaled_data


def apply_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    pca_result = pd.DataFrame(pca.fit_transform(data), columns=[f"col{i}" for i in range(1, n_components + 1)])
    return pca_result


def preprocess_data():
    data = parse_data()
    data = drop_columns(data)
    data = data.dropna()
    data = encode_categorical(data)
    scaled_data = scale_data(data)
    pca_result = apply_pca(scaled_data)
    return pca_result




