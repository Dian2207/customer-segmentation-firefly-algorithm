import pandas as pd
import numpy as np

from ml.ga import run_ga
from ml.kmeans import run_kmeans


def compute_wcss(data, labels, centroids):

    wcss = 0

    for i in range(len(centroids)):

        cluster_points = data[labels == i]

        if len(cluster_points) > 0:

            wcss += np.sum(
                (cluster_points - centroids[i]) ** 2
            )

    return wcss

def find_best_k_elbow(
    data,
    k_range=range(2, 10)
):

    print("\nSTEP GA: MENCARI K TERBAIK")

    wcss_values = []

    for k in k_range:

        print(f"\n--- Testing K = {k} ---")

        centroids = run_ga(data, k)

        # validasi centroid
        if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):

            print("WARNING: centroid invalid → random init")

            idx = np.random.choice(
                len(data),
                k,
                replace=False
            )

            centroids = data[idx]

        labels, centroids = run_kmeans(
            data,
            centroids
        )

        wcss = compute_wcss(
            data,
            labels,
            centroids
        )

        wcss_values.append(wcss)

        print("WCSS:", round(wcss, 2))

    deltas = np.diff(wcss_values)

    best_k = k_range[
        np.argmin(deltas) + 1
    ]

    print("\nELBOW ditemukan pada K =", best_k)

    return best_k, wcss_values

def run_ga_clustering():

    print("\nSTEP GA CLUSTERING")

    df = pd.read_csv(
        "data/processed/train.csv"
    )

    data = df.values

    print("Jumlah data:", data.shape)

    best_k, wcss_values = find_best_k_elbow(data)

    print("\nK terbaik:", best_k)

    print("\nTraining final clustering...")

    centroids = run_ga(
        data,
        best_k
    )

    if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):

        print("WARNING: centroid invalid → random init")

        idx = np.random.choice(
            len(data),
            best_k,
            replace=False
        )

        centroids = data[idx]

    labels, centroids = run_kmeans(
        data,
        centroids
    )

    df["Cluster"] = labels

    df.to_csv(
        "data/processed/clustered_ga.csv",
        index=False
    )

    print("\nGA clustering selesai ")

    print("File tersimpan:")
    print("data/processed/clustered_ga.csv")

    return labels, centroids, best_k