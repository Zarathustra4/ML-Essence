import matplotlib.pyplot as plt
from kaggle_sets.clustering.model import Clusterer
from kaggle_sets.clustering.data_preparation import preprocess_data


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

        self.plot_3d_clusters(preprocessed_data['x'], preprocessed_data['y'], preprocessed_data['z'])

if __name__ == "__main__":
    cluster_service = ClustererService(n_clusters=2)
    cluster_service.run_cluster_analysis()
