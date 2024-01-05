from kaggle_sets.model_service.clusterer_service import ClustererService


def train_save_clusterer():
    cluster_service = ClustererService(n_clusters=3)
    preprocessed_data = cluster_service.run_cluster_analysis()

    print("| --- Cluster Analysis --- |")
    print(f"Clusters: {cluster_service.get_clusters()}")

    cluster_service.plot_3d_clusters(preprocessed_data['x'], preprocessed_data['y'], preprocessed_data['z'])

if __name__ == "__main__":
    train_save_clusterer()