from sklearn.cluster import AgglomerativeClustering
from kaggle_sets.clustering.data_preparation import preprocess_data
import matplotlib.pyplot as plt


class Clusterer:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.data = None
        self.cluster_model = AgglomerativeClustering(n_clusters=self.n_clusters)

    def fit_predict_clusters(self, preprocessed_data):
        self.data = preprocessed_data
        clusters = self.cluster_model.fit_predict(self.data)
        self.data["Clusters"] = clusters
        return self.data

    def get_clusters(self):
        return self.data["Clusters"]

    def plot_3d_clusters(self, x, y, z):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = 'viridis'

        scatter = ax.scatter(x, y, z, s=40, c=self.data["Clusters"], marker='o', cmap=cmap)

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Cluster')

        ax.set_title("The Plot Of The Clusters")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        plt.show()
