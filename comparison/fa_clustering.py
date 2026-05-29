import pandas as pd
import numpy as np

from ml.firefly import run_firefly
from ml.hybrid_clustering import compute_wcss
from sklearn.metrics import silhouette_score


def predict_cluster(data, centroids):

    distances = np.linalg.norm(
        data[:, None] - centroids,
        axis=2
    )

    return np.argmin(
        distances,
        axis=1
    )


def run_fa_clustering():

    print("\nSTEP 5: TRAIN CLUSTERING FIREFLY")

    df = pd.read_csv(
        "data/processed/train.csv"
    )

    data = df.values

    print("Jumlah data:", data.shape)

    best_k = 2
    best_silhouette = -1

    best_labels = None
    best_centroids = None

    iteration_results = []

    for k in range(2, 10):

        print(f"\n--- Testing K = {k} ---")

        centroids = run_firefly(
            data,
            k
        )

        labels = predict_cluster(
            data,
            centroids
        )

        sse = compute_wcss(
            data,
            labels,
            centroids
        )

        if len(np.unique(labels)) < 2:

            silhouette = -1

        else:

            silhouette = silhouette_score(
        data,
        labels
    )

        print(
    f"SSE = {round(sse,2)} | "
    f"Silhouette = {silhouette:.4f}"
)

        iteration_results.append({

    "k": k,
    "sse": sse,
    "silhouette": silhouette

})

        if silhouette > best_silhouette:

            best_silhouette = silhouette

            best_k = k

            best_labels = labels

            best_centroids = centroids

    print("\nK terbaik:", best_k)

    df["Cluster"] = best_labels

    df.to_csv(
        "data/result/clustered_train_firefly.csv",
        index=False
    )

    print(
        "\nFile tersimpan:"
        " data/result/clustered_train_firefly.csv"
    )

    return (
        best_labels,
        best_centroids,
        best_k,
        iteration_results
    )