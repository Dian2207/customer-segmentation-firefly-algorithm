import time
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

from ml.hybrid_clustering import (
    compute_wcss
)

from ml.firefly import run_firefly
from ml.kmeans import run_kmeans

def predict_cluster(data, centroids):

    distances = np.linalg.norm(
        data[:, None] - centroids,
        axis=2
    )

    return np.argmin(
        distances,
        axis=1
    )

def evaluate_hybrid_firefly():

    print("\n================================")
    print("EVALUASI HYBRID FIREFLY")
    print("================================")

    start_time = time.time()

    val = pd.read_csv(
        "data/processed/validation.csv"
    )

    data = val.values

    results = []

    best_score = -1
    best_k = 2
    best_labels = None
    best_centroids = None

    for k in range(2, 10):

        print(f"\n--- Testing K = {k} ---")

        iter_start = time.time()

        init_idx = np.random.choice(
            len(data),
            k,
            replace=False
        )

        init_centroids = data[init_idx]

        labels, centroids = run_kmeans(
            data,
            init_centroids
        )

        optimized_centroids = run_firefly(
            data,
            k,
            init_centroids=centroids
        )

        labels, final_centroids = run_kmeans(
            data,
            optimized_centroids
        )

        if len(np.unique(labels)) < 2:

            silhouette = -1

        else:

            silhouette = silhouette_score(
                data,
                labels
            )

        sse = compute_wcss(
            data,
            labels,
            final_centroids
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

        if silhouette > best_score:

            best_score = silhouette

            best_k = k

            best_labels = labels

            best_centroids = final_centroids

    end_time = time.time()

    execution_time = end_time - start_time

    print("\nSilhouette Score :", best_score)

    print(
        "Execution Time   :",
        round(execution_time, 4),
        "detik"
    )

    return {

        "method": "Hybrid KMeans-Firefly",

        "silhouette": best_score,

        "time": execution_time,

        "best_k": best_k,

        "labels": best_labels,

        "centroids": best_centroids,

        "iterations": results
    }

def evaluate_firefly_only():

    print("\n================================")
    print("EVALUASI FIREFLY ONLY")
    print("================================")

    start_time = time.time()

    val = pd.read_csv(
        "data/processed/validation.csv"
    )

    data = val.values

    results = []

    best_score = -1
    best_k = 2
    best_labels = None
    best_centroids = None

    for k in range(2, 10):

        print(f"\n--- Testing K = {k} ---")

        iter_start = time.time()

        centroids = run_firefly(
            data,
            k
        )

        labels = predict_cluster(
            data,
            centroids
        )

        if len(np.unique(labels)) < 2:

            silhouette = -1

        else:

            silhouette = silhouette_score(
                data,
                labels
            )

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

        if silhouette > best_score:

            best_score = silhouette

            best_k = k

            best_labels = labels

            best_centroids = centroids

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

        "centroids": best_centroids,

        "iterations": results
    }

def run_firefly_comparison():

    print("\n======================================")
    print("PERBANDINGAN METODE CLUSTERING")
    print("======================================")

    hybrid_result = evaluate_hybrid_firefly()

    firefly_result = evaluate_firefly_only()

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

    print("\nHASIL PERBANDINGAN")

    print(comparison)

    comparison.to_csv(

        "data/result/comparison_firefly_result.csv",

        index=False

    )

    print("\nFile tersimpan:")

    print(
        "data/result/comparison_firefly_result.csv"
    )

    pd.DataFrame(
        firefly_result["iterations"]
    ).to_csv(

        "data/result/firefly_iterations.csv",

        index=False

    )

    print(
        "data/result/firefly_iterations.csv"
    )

    val_df = pd.read_csv(
        "data/processed/validation.csv"
    )

    hybrid_cluster = val_df.copy()

    hybrid_cluster["Cluster"] = (
        hybrid_result["labels"]
    )

    hybrid_cluster.to_csv(

        "data/result/clustered_hybrid_firefly.csv",

        index=False

    )

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
