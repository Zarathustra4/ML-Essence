from kaggle_sets.model_service.clusterer_service import ClustererService


def train_save_clusterer():
    cluster_service = ClustererService(n_clusters=2)
    preprocessed_data = cluster_service.run_cluster_analysis()

    print("| --- Cluster Analysis --- |")
    print(f"Clusters: {cluster_service.get_clusters()}")


if __name__ == "__main__":
    train_save_clusterer()