import os.path

import matplotlib.pyplot as plt
import pandas as pd
import kaggle_sets.config as conf
from kaggle_sets.clustering.model import Clusterer
from kaggle_sets.clustering.data_preparation import preprocess_data, drop_columns, encode_categorical, scale_data, apply_pca


class ClustererService:
    def __init__(self, n_clusters=2):
        self.clusterer = Clusterer(n_clusters)

    def cluster_data(self, preprocessed_data):
        """
        Fits and predicts clusters on preprocessed data
        :param preprocessed_data: Input preprocessed data
        :return: DataFrame with cluster labels
        """
        clustered_data = self.clusterer.fit_predict_clusters(preprocessed_data)
        return clustered_data

    def cluster_data_by_csv(self, filename, delimiter=","):
        filename = os.path.join(conf.BASE_DATASET_PATH, filename + ".csv")
        data = pd.read_csv(filename, delimiter=delimiter)
        data = drop_columns(data)
        data = data.dropna()
        data = encode_categorical(data)
        scaled_data = scale_data(data)
        pca_result = apply_pca(scaled_data)

        clustered_data = self.clusterer.fit_predict_clusters(pca_result)
        return clustered_data

    def get_clusters(self):
        """
        Get cluster labels
        :return: Cluster labels
        """
        return self.clusterer.get_clusters()

    def plot_3d_clusters(self, x, y, z):
        """
        Plot 3D clusters
        :param x: X-axis values
        :param y: Y-axis values
        :param z: Z-axis values
        :return: None
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = 'viridis'

        scatter = ax.scatter(x, y, z, s=40, c=self.clusterer.get_clusters(), marker='o', cmap=cmap)

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Cluster')

        ax.set_title("The Plot Of The Clusters")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        plt.show()

    def run_cluster_analysis(self):
        preprocessed_data = preprocess_data()

        clustered_data = self.cluster_data(preprocessed_data)

        clusters = self.get_clusters()
        print(f"Clusters: {clusters}")

        self.plot_3d_clusters(clustered_data['col1'], clustered_data['col2'], clustered_data['col3'])
        return clustered_data


def train_save_clusterer():
    cluster_service = ClustererService(n_clusters=2)
    preprocessed_data = cluster_service.run_cluster_analysis()

    print("| --- Cluster Analysis --- |")
    print(f"Clusters: {cluster_service.get_clusters()}")

