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

def compute_wcss(data, labels, centroids):
    wcss = 0

    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]

        if len(cluster_points) > 0:
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            wcss += np.sum(distances ** 2)

    return wcss

def elbow_method(data, k_range=range(2, 10), max_iter=50):
    wcss_values = []

    print("Menjalankan Elbow Method...")

    for k in k_range:

        init_centroids = data[np.random.choice(len(data), k, replace=False)]

        labels, centroids = run_kmeans(data, init_centroids, max_iter)

        wcss = compute_wcss(data, labels, centroids)
        wcss_values.append(wcss)

        print(f"K = {k} | WCSS = {wcss:.2f}")

    return list(k_range), wcss_values