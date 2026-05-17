import pandas as pd
import numpy as np
from ml.firefly import run_firefly
from ml.kmeans import run_kmeans

def compute_wcss(data, labels, centroids):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss

def find_best_k_elbow(data, k_range=range(2, 10)):
    print("\nSTEP 5A: MENCARI K TERBAIK (ELBOW METHOD)")
    wcss_values = []

    for k in k_range:
        print(f"\n--- Testing K = {k} ---")

        centroids = run_firefly(data, k)

        if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):
            print("WARNING: centroid invalid → random init")
            idx = np.random.choice(len(data), k, replace=False)
            centroids = data[idx]

        labels, centroids = run_kmeans(data, centroids)

        wcss = compute_wcss(data, labels, centroids)
        wcss_values.append(wcss)

        print("WCSS:", round(wcss, 2))

    deltas = np.diff(wcss_values)
    best_k = k_range[np.argmin(deltas) + 1]

    print("\nELBOW ditemukan pada K =", best_k)
    return best_k, wcss_values

def run_hybrid_clustering():
    print("\nSTEP 5: TRAIN CLUSTERING HYBRID")

    df = pd.read_csv("data/processed/train.csv")
    data = df.values
    print("Jumlah data:", data.shape)

    best_k, wcss_values = find_best_k_elbow(data)
    print("\nK terbaik hasil Elbow:", best_k)

    print("\nTraining final clustering...")

    centroids = run_firefly(data, best_k)

    if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):
        print("WARNING: centroid invalid → random init")
        idx = np.random.choice(len(data), best_k, replace=False)
        centroids = data[idx]

    labels, centroids = run_kmeans(data, centroids)

    df["Cluster"] = labels
    df.to_csv("data/processed/clustered_train.csv", index=False)

    print("\nClustering selesai 🚀")
    print("File tersimpan: data/processed/clustered_train.csv")

    return centroids, best_k