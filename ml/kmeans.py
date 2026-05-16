# ml/kmeans_custom.py
import numpy as np

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(np.random.rand(data.shape[1]))
    return np.array(new_centroids)

def run_kmeans(data, init_centroids, max_iter=50):
    centroids = init_centroids

    for _ in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, len(centroids))

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids