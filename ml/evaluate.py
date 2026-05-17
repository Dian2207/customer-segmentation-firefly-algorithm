import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from ml.hybrid_clustering import run_hybrid_clustering

def predict_cluster(data, centroids):
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def run_full_evaluation():
    print("STEP 5: HYBRID & EVALUATION CLUSTERING")

    centroids, best_k = run_hybrid_clustering()

    train = pd.read_csv("data/processed/train.csv").values
    val   = pd.read_csv("data/processed/validation.csv").values
    test  = pd.read_csv("data/processed/test.csv").values

    print("\nJumlah cluster terbaik:", best_k)

    labels_val = predict_cluster(val, centroids)

    print("\nDistribusi cluster (validation):")
    print(pd.Series(labels_val).value_counts())

    if len(np.unique(labels_val)) < 2:
        print("WARNING: hanya 1 cluster ditemukan -> silhouette dilewati")
        score = -1
    else:
        silhouette = silhouette_score(val, labels_val)

    print("\nSilhouette Score (Validation):", silhouette)

    labels_test = predict_cluster(test, centroids)

    df_test = pd.read_csv("data/processed/test.csv")
    df_test["Cluster"] = labels_test
    df_test.to_csv("data/processed/clustered_test.csv", index=False)

    print("Hasil clustering test disimpan ke clustered_test.csv")

    return silhouette, best_k