# ==========================================
# PARAMETER_STABILITY_COMPARISON.PY
# ==========================================

import time
import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score

from comparison.kmeans_clustering import (
    run_kmeans_clustering
)

from comparison.ga_clustering import (
    run_ga_clustering
)

from ml.hybrid_clustering import (
    run_hybrid_clustering
)

from comparison.evaluate_firefly_comparison import (
    evaluate_firefly_only
)


# ==========================================
# PREDICT CLUSTER
# ==========================================
def predict_cluster(data, centroids):

    distances = np.linalg.norm(
        data[:, None] - centroids,
        axis=2
    )

    return np.argmin(
        distances,
        axis=1
    )


# ==========================================
# LOAD VALIDATION
# ==========================================
val = pd.read_csv(
    "data/processed/validation.csv"
).values


results = []


# ==========================================
# ITERASI 50X
# ==========================================
for iteration in range(1, 51):

    print(f"\n================================")
    print(f"ITERASI {iteration}")
    print("================================")

    np.random.seed(None)

    # ======================================
    # KMEANS
    # ======================================
    start = time.time()

    labels_train, centroids, best_k = (
        run_kmeans_clustering()
    )

    execution_time = time.time() - start

    labels = predict_cluster(
        val,
        centroids
    )

    silhouette = silhouette_score(
        val,
        labels
    )

    results.append({

        "iteration": iteration,

        "kmeans_silhouette": silhouette,

        "kmeans_time": execution_time

    })


    # ======================================
    # FIREFLY ONLY
    # ======================================
    start = time.time()

    firefly_result = evaluate_firefly_only()

    execution_time = time.time() - start

    results[-1]["firefly_silhouette"] = (
        firefly_result["silhouette"]
    )

    results[-1]["firefly_time"] = (
        execution_time
    )


    # ======================================
    # HYBRID KMEANS + FIREFLY
    # ======================================
    start = time.time()

    centroids, best_k = (
        run_hybrid_clustering()
    )

    execution_time = time.time() - start

    labels = predict_cluster(
        val,
        centroids
    )

    silhouette = silhouette_score(
        val,
        labels
    )

    results[-1][
        "hybrid_firefly_silhouette"
    ] = silhouette

    results[-1][
        "hybrid_firefly_time"
    ] = execution_time


    # ======================================
    # HYBRID KMEANS + GA
    # ======================================
    start = time.time()

    labels_train, centroids, best_k = (
        run_ga_clustering()
    )

    execution_time = time.time() - start

    labels = predict_cluster(
        val,
        centroids
    )

    silhouette = silhouette_score(
        val,
        labels
    )

    results[-1][
        "hybrid_ga_silhouette"
    ] = silhouette

    results[-1][
        "hybrid_ga_time"
    ] = execution_time


# ==========================================
# SAVE CSV
# ==========================================
df = pd.DataFrame(results)

df.to_csv(

    "data/result/parameter_stability.csv",

    index=False

)

print("\n================================")
print("FILE BERHASIL DISIMPAN")
print("================================")

print(
    "data/result/parameter_stability.csv"
)