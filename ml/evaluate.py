import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score

from ml.hybrid_clustering import run_hybrid_clustering


def predict_cluster(data, centroids):

    distances = np.linalg.norm(
        data[:, None] - centroids,
        axis=2
    )

    return np.argmin(distances, axis=1)


def run_full_evaluation():

    print("STEP 6: EVALUATION")

    centroids, best_k = run_hybrid_clustering()

    train = pd.read_csv(
        "data/processed/train.csv"
    ).values

    val = pd.read_csv(
        "data/processed/validation.csv"
    ).values

    test = pd.read_csv(
        "data/processed/test.csv"
    ).values

    print("\nJumlah cluster terbaik:", best_k)

    labels_val = predict_cluster(val, centroids)

    print("\nDistribusi cluster validation:")
    print(pd.Series(labels_val).value_counts())

    if len(np.unique(labels_val)) < 2:

        silhouette = -1

        print(
            "WARNING: hanya 1 cluster ditemukan"
        )

    else:

        silhouette = silhouette_score(
            val,
            labels_val
        )

    print(
        "\nSilhouette Score:",
        round(silhouette, 4)
    )

    labels_test = predict_cluster(
        test,
        centroids
    )

    df_test = pd.read_csv(
        "data/processed/test.csv"
    )

    df_test["Cluster"] = labels_test

    df_test.to_csv(
        "data/processed/clustered_test.csv",
        index=False
    )

    print(
        "\nHasil clustering test disimpan"
    )

    df_test_raw = pd.read_csv(
        "data/processed/test_raw.csv"
    ).reset_index(drop=True)

    min_len = min(
        len(df_test_raw),
        len(labels_test)
    )

    df_test_raw = df_test_raw.iloc[:min_len].copy()

    labels_test = labels_test[:min_len]

    df_test_raw["Cluster"] = labels_test

    cluster_summary = (
        df_test_raw
        .groupby("Cluster")[[
            "Recency",
            "Frequency",
            "Monetary"
        ]]
        .mean()
        .reset_index()
        .sort_values("Cluster")
    )

    cluster_summary.to_csv(
        "data/result/cluster_summary.csv",
        index=False
    )

    print("\nCluster Summary:")
    print(cluster_summary)

    return silhouette, best_k, cluster_summary
