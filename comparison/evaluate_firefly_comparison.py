import time
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

from ml.hybrid_clustering import (
    run_hybrid_clustering,
    compute_wcss
)

from ml.firefly import run_firefly
from ml.kmeans import run_kmeans


# ==================================================
# PREDICT CLUSTER
# ==================================================
def predict_cluster(data, centroids):

    distances = np.linalg.norm(
        data[:, None] - centroids,
        axis=2
    )

    return np.argmin(
        distances,
        axis=1
    )


# ==================================================
# EVALUASI HYBRID FIREFLY
# ==================================================
def evaluate_hybrid_firefly():

    print("\n================================")
    print("EVALUASI HYBRID FIREFLY")
    print("================================")

    start_time = time.time()

    # ==============================================
    # RUN HYBRID
    # ==============================================
    centroids, best_k = run_hybrid_clustering()

    end_time = time.time()

    execution_time = end_time - start_time

    # ==============================================
    # VALIDATION DATA
    # ==============================================
    val = pd.read_csv(
        "data/processed/validation.csv"
    ).values

    labels = predict_cluster(
        val,
        centroids
    )

    # ==============================================
    # SILHOUETTE
    # ==============================================
    if len(np.unique(labels)) < 2:

        silhouette = -1

        print("WARNING: hanya 1 cluster")

    else:

        silhouette = silhouette_score(
            val,
            labels
        )

    print("\nSilhouette Score :", silhouette)

    print(
        "Execution Time   :",
        round(execution_time, 4),
        "detik"
    )

    return {

        "method": "Hybrid KMeans-Firefly",

        "silhouette": silhouette,

        "time": execution_time,

        "best_k": best_k,

        "labels": labels
    }


# ==================================================
# EVALUASI FIREFLY ONLY
# ==================================================
def evaluate_firefly_only():

    print("\n================================")
    print("EVALUASI FIREFLY ONLY")
    print("================================")

    start_time = time.time()

    # ==============================================
    # LOAD DATA
    # ==============================================
    val = pd.read_csv(
        "data/processed/validation.csv"
    )

    data = val.values

    results = []

    best_score = -1
    best_k = 2
    best_labels = None

    # ==============================================
    # ITERASI K
    # ==============================================
    for k in range(2, 10):

        print(f"\n--- Testing K = {k} ---")

        iter_start = time.time()

        # ==========================================
        # RUN FIREFLY
        # ==========================================
        centroids = run_firefly(
            data,
            k
        )

        # ==========================================
        # LABEL
        # ==========================================
        labels = predict_cluster(
            data,
            centroids
        )

        # ==========================================
        # SILHOUETTE
        # ==========================================
        if len(np.unique(labels)) < 2:

            silhouette = -1

        else:

            silhouette = silhouette_score(
                data,
                labels
            )

        # ==========================================
        # SSE
        # ==========================================
        sse = compute_wcss(
            data,
            labels,
            centroids
        )

        iter_time = time.time() - iter_start

        print(
            f"K = {k} | "
            f"Silhouette = {silhouette:.4f} | "
            f"SSE = {sse:.2f} | "
            f"Time = {iter_time:.4f}"
        )

        results.append({

            "k": k,

            "silhouette": silhouette,

            "sse": sse,

            "time": iter_time
        })

        # ==========================================
        # BEST RESULT
        # ==========================================
        if silhouette > best_score:

            best_score = silhouette

            best_k = k

            best_labels = labels

    end_time = time.time()

    execution_time = end_time - start_time

    print("\nSilhouette Score :", best_score)

    print(
        "Execution Time   :",
        round(execution_time, 4),
        "detik"
    )

    return {

        "method": "Firefly Only",

        "silhouette": best_score,

        "time": execution_time,

        "best_k": best_k,

        "labels": best_labels,

        "iterations": results
    }


# ==================================================
# MAIN COMPARISON
# ==================================================
def run_firefly_comparison():

    print("\n======================================")
    print("PERBANDINGAN METODE CLUSTERING")
    print("======================================")

    # ==============================================
    # HYBRID
    # ==============================================
    hybrid_result = evaluate_hybrid_firefly()

    # ==============================================
    # FIREFLY ONLY
    # ==============================================
    firefly_result = evaluate_firefly_only()

    # ==============================================
    # COMPARISON TABLE
    # ==============================================
    comparison = pd.DataFrame([

        {

            "method": hybrid_result["method"],

            "silhouette": hybrid_result["silhouette"],

            "time": hybrid_result["time"],

            "best_k": hybrid_result["best_k"]

        },

        {

            "method": firefly_result["method"],

            "silhouette": firefly_result["silhouette"],

            "time": firefly_result["time"],

            "best_k": firefly_result["best_k"]

        }

    ])

    # ==============================================
    # PRINT RESULT
    # ==============================================
    print("\nHASIL PERBANDINGAN")

    print(comparison)

    # ==============================================
    # SAVE CSV
    # ==============================================
    comparison.to_csv(

        "data/result/comparison_firefly_result.csv",

        index=False

    )

    print("\nFile tersimpan:")

    print(
        "data/result/comparison_firefly_result.csv"
    )

    # ==============================================
    # SAVE ITERATION CSV
    # ==============================================
    pd.DataFrame(
        firefly_result["iterations"]
    ).to_csv(

        "data/result/firefly_iterations.csv",

        index=False

    )

    print(
        "data/result/firefly_iterations.csv"
    )

    # ==============================================
    # SAVE CLUSTER RESULT
    # ==============================================
    val_df = pd.read_csv(
        "data/processed/validation.csv"
    )

    # HYBRID
    hybrid_cluster = val_df.copy()

    hybrid_cluster["Cluster"] = (
        hybrid_result["labels"]
    )

    hybrid_cluster.to_csv(

        "data/result/clustered_hybrid_firefly.csv",

        index=False

    )

    # FIREFLY
    firefly_cluster = val_df.copy()

    firefly_cluster["Cluster"] = (
        firefly_result["labels"]
    )

    firefly_cluster.to_csv(

        "data/result/clustered_firefly_only.csv",

        index=False

    )

    print(
        "data/result/clustered_hybrid_firefly.csv"
    )

    print(
        "data/result/clustered_firefly_only.csv"
    )

    return comparison