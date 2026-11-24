import matplotlib.pyplot as plt

from MLToolKit.toolkit_loader import load_impl


def test_clustering_methods_assign_every_row():
    data = load_impl("implementations/09_Utilities/synthetic_datasets.py")
    kmeans = load_impl("implementations/05_Clustering/kmeans.py")
    hierarchical = load_impl("implementations/05_Clustering/hierarchical.py")
    dbscan = load_impl("implementations/05_Clustering/dbscan.py")
    elbow = load_impl("implementations/05_Clustering/elbow_method.py")
    silhouette = load_impl("implementations/05_Clustering/silhouette_analysis.py")
    plots = load_impl("implementations/05_Clustering/visualization.py")
    frame = data.cluster_frame(n_samples=90)
    features = frame[["feature_0", "feature_1"]]

    kmeans_result = kmeans.train_kmeans(features, n_clusters=3)
    hierarchical_result = hierarchical.train_hierarchical(features, n_clusters=3)
    dbscan_result = dbscan.train_dbscan(features, eps=0.6, min_samples=4)
    assert len(kmeans_result["labels"]) == len(features)
    assert len(hierarchical_result["labels"]) == len(features)
    assert len(dbscan_result["labels"]) == len(features)
    inertias = elbow.elbow_inertias(features, range(1, 5))
    assert inertias["inertia"].iloc[0] > inertias["inertia"].iloc[-1]
    assert silhouette.best_silhouette_k(features, range(2, 5)) in {2, 3, 4}

    fig, _ax = plots.plot_clusters(kmeans_result["X_scaled"], kmeans_result["labels"])
    plt.close(fig)
    fig, _ax = hierarchical.plot_dendrogram(features.iloc[:25])
    plt.close(fig)
