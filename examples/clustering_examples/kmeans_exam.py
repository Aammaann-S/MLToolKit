"""Short K-Means practical on synthetic blobs."""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _labels = make_blobs(n_samples=120, centers=3, random_state=42)
X_scaled = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = model.fit_predict(X_scaled)
print("clusters", len(set(clusters)))
