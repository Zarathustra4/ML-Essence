from sklearn.cluster import AgglomerativeClustering
from clustering.data_preparation import preprocess_data
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

