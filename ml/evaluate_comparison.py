import time
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

from ml.hybrid_clustering import run_hybrid_clustering
from ml.ga_clustering import run_ga_clustering

def predict_cluster(data, centroids):

    distances = np.linalg.norm(
        data[:, None] - centroids,
        axis=2
    )

    return np.argmin(
        distances,
        axis=1
    )


def evaluate_firefly():

    print("\n================================")
    print("EVALUASI HYBRID FIREFLY")
    print("================================")

    start_time = time.time()

    centroids, best_k = run_hybrid_clustering()

    end_time = time.time()

    execution_time = end_time - start_time

    val = pd.read_csv(
        "data/processed/validation.csv"
    ).values

    labels = predict_cluster(
        val,
        centroids
    )

    if len(np.unique(labels)) < 2:

        silhouette = -1

        print("WARNING: hanya 1 cluster")

    else:

        silhouette = silhouette_score(
            val,
            labels
        )

    print("\nSilhouette Score :", silhouette)

    print("Execution Time   :", round(execution_time, 4), "detik")

    return {

        "method": "Hybrid Firefly",

        "silhouette": silhouette,

        "time": execution_time,

        "best_k": best_k

    }

def evaluate_ga():

    print("\n================================")
    print("EVALUASI HYBRID GA")
    print("================================")

    start_time = time.time()

    labels_train, centroids, best_k = run_ga_clustering()

    end_time = time.time()

    execution_time = end_time - start_time

    val = pd.read_csv(
        "data/processed/validation.csv"
    ).values

    labels = predict_cluster(
        val,
        centroids
    )

    if len(np.unique(labels)) < 2:

        silhouette = -1

        print("WARNING: hanya 1 cluster")

    else:

        silhouette = silhouette_score(
            val,
            labels
        )

    print("\nSilhouette Score :", silhouette)

    print("Execution Time   :", round(execution_time, 4), "detik")

    return {

        "method": "Hybrid GA",

        "silhouette": silhouette,

        "time": execution_time,

        "best_k": best_k

    }

def run_comparison():

    print("\n======================================")
    print("PERBANDINGAN METODE CLUSTERING")
    print("======================================")

    firefly_result = evaluate_firefly()

    ga_result = evaluate_ga()

    comparison = pd.DataFrame([

        firefly_result,
        ga_result

    ])

    print("\nHASIL PERBANDINGAN")
    print(comparison)

    comparison.to_csv(

        "data/processed/comparison_result.csv",

        index=False

    )

    print("\nFile tersimpan:")
    print("data/processed/comparison_result.csv")

    return comparison