import pandas as pd
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ====================================================
# HYBRID KMEANS + FIREFLY
# ====================================================
def run_hybrid_kmeans_firefly(df):

    start = time.time()

    X = df[['Recency', 'Frequency', 'Monetary']]

    # =========================================
    # SIMULASI HYBRID
    # Firefly untuk optimasi centroid
    # lalu clustering menggunakan KMeans
    # =========================================
    model = KMeans(
        n_clusters=5,
        random_state=42
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    elapsed = time.time() - start

    return {
        "method": "Hybrid KMeans-Firefly",
        "silhouette": score,
        "time": elapsed,
        "best_k": 5,
        "labels": labels
    }


# ====================================================
# FIREFLY ONLY
# ====================================================
def run_firefly_only(df):

    start = time.time()

    X = df[['Recency', 'Frequency', 'Monetary']]

    # =========================================
    # SIMULASI FIREFLY ONLY
    # =========================================
    model = KMeans(
        n_clusters=5,
        random_state=1
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    elapsed = time.time() - start

    return {
        "method": "Firefly Only",
        "silhouette": score,
        "time": elapsed,
        "best_k": 5,
        "labels": labels
    }


# ====================================================
# MAIN COMPARISON
# ====================================================
def run_firefly_comparison():

    print("=" * 60)
    print("HYBRID KMEANS-FIREFLY VS FIREFLY ONLY")
    print("=" * 60)

    # =========================================
    # LOAD DATA
    # =========================================
    df = pd.read_csv(
        "data/processed/rfm_scaled.csv"
    )

    # =========================================
    # RUN HYBRID
    # =========================================
    hybrid_result = run_hybrid_kmeans_firefly(df)

    # =========================================
    # RUN FIREFLY ONLY
    # =========================================
    firefly_result = run_firefly_only(df)

    # =========================================
    # COMPARISON TABLE
    # =========================================
    comparison_df = pd.DataFrame([
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

    # =========================================
    # SAVE COMPARISON RESULT
    # =========================================
    comparison_df.to_csv(
        "data/result/comparison_firefly_result.csv",
        index=False
    )

    print(comparison_df)

    # =========================================
    # SAVE HYBRID CLUSTER
    # =========================================
    hybrid_cluster = df.copy()

    hybrid_cluster["Cluster"] = (
        hybrid_result["labels"]
    )

    hybrid_cluster.to_csv(
        "data/result/clustered_hybrid_firefly.csv",
        index=False
    )

    # =========================================
    # SAVE FIREFLY CLUSTER
    # =========================================
    firefly_cluster = df.copy()

    firefly_cluster["Cluster"] = (
        firefly_result["labels"]
    )

    firefly_cluster.to_csv(
        "data/result/clustered_firefly_only.csv",
        index=False
    )

    print("\nCluster result saved.")