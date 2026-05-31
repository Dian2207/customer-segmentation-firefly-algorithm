import pandas as pd
import numpy as np

from ml.kmeans import run_kmeans
from ml.firefly import run_firefly

np.random.seed(None)


def compute_wcss(data, labels, centroids):

    wcss = 0

    for i in range(len(centroids)):

        cluster_points = data[labels == i]

        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i]) ** 2)

    return wcss


def hybrid_kmeans_firefly(data, k):

    init_idx = np.random.choice(len(data), k, replace=False)
    init_centroids = data[init_idx]

    labels, centroids = run_kmeans(
        data,
        init_centroids
    )
    optimized_centroids = run_firefly(
        data,
        k,
        init_centroids=centroids,
        n_fireflies=10,
        max_iter=30
    )

    final_labels, final_centroids = run_kmeans(
        data,
        optimized_centroids
    )

    return final_labels, final_centroids


def find_best_k_elbow(data, k_range=range(2, 10)):

    print("\nSTEP 5A: ELBOW METHOD")

    wcss_values = []

    for k in k_range:

        print(f"\nTesting K = {k}")

        labels, centroids = hybrid_kmeans_firefly(data, k)

        wcss = compute_wcss(data, labels, centroids)

        wcss_values.append(wcss)

        print("WCSS:", round(wcss, 2))

    k_values = list(k_range)

    points = np.column_stack((k_values, wcss_values))

    first_point = points[0]
    last_point = points[-1]

    line_vector = last_point - first_point
    line_vector = line_vector / np.linalg.norm(line_vector)

    distances = []

    for point in points:

        vector = point - first_point

        projection = np.dot(vector, line_vector) * line_vector

        perpendicular = vector - projection

        distances.append(np.linalg.norm(perpendicular))

    best_index = np.argmax(distances)

    best_k = k_values[best_index]

    print("\nBest K:", best_k)

    return best_k, wcss_values


def run_hybrid_clustering():

    print("\nSTEP 5: HYBRID CLUSTERING")

    df = pd.read_csv("data/processed/train.csv")

    data = df.values

    print("Jumlah data:", data.shape)

    best_k, wcss_values = find_best_k_elbow(data)

    print("\nK terbaik:", best_k)

    labels, centroids = hybrid_kmeans_firefly(
        data,
        best_k
    )

    df["Cluster"] = labels

    df.to_csv(
        "data/processed/clustered_train.csv",
        index=False
    )

    print("\nClustering selesai")

    return centroids, best_k
