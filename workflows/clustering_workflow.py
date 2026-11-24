"""Cluster synthetic blobs and report the silhouette score."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MLToolKit.toolkit_loader import load_impl


def main():
    data = load_impl("implementations/09_Utilities/synthetic_datasets.py")
    kmeans = load_impl("implementations/05_Clustering/kmeans.py")
    silhouette = load_impl("implementations/05_Clustering/silhouette_analysis.py")
    frame = data.cluster_frame()
    features = frame.drop(columns=["cluster"])
    result = kmeans.train_kmeans(features, n_clusters=3)
    scores = silhouette.silhouette_scores(features, k_values=[2, 3, 4])
    best = float(scores["silhouette"].max())
    print("cluster_rows", len(result["labels"]))
    print("best_silhouette", round(best, 4))
    if best < 0.2:
        raise SystemExit("Clustering workflow silhouette was below 0.2")
    return result


if __name__ == "__main__":
    main()
