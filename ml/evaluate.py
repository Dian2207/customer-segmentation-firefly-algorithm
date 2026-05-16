import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from ml.firefly import run_firefly
from ml.kmeans import run_kmeans

def predict_cluster(data, centroids):
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def run_full_evaluation():
    print("STEP 5: TRAIN & EVALUATION CLUSTERING")

    train = pd.read_csv("data/processed/train.csv").values
    val   = pd.read_csv("data/processed/validation.csv").values
    test  = pd.read_csv("data/processed/test.csv").values

    best_k = 0
    best_score = -1
    best_centroids = None

    for k in [2, 3, 4, 5, 6]:
        print(f"\n--- Testing K = {k} ---")

        # =========================
        # HYBRID FIRELY + KMEANS
        # =========================
        centroids = run_firefly(train, k)
        labels_train, centroids = run_kmeans(train, centroids)

        # =========================
        # VALIDATION PREDICTION
        # =========================
        labels_val = predict_cluster(val, centroids)

        # =========================
        # DEBUG CLUSTER DISTRIBUTION
        # =========================
        unique_labels = len(set(labels_val))
        print("Cluster distribution (val):")
        print(pd.Series(labels_val).value_counts())

        # =========================
        # SAFE SILHOUETTE SCORE
        # =========================
        if unique_labels < 2:
            print("WARNING: hanya 1 cluster ditemukan -> silhouette dilewati")
            score = -1
        else:
            print("UNIQUE CLUSTERS:", np.unique(labels_val))
            print("COUNT:", len(np.unique(labels_val)))
            print(pd.Series(labels_val).value_counts())
            score = silhouette_score(val, labels_val)

        print("Silhouette Score:", score)

        # =========================
        # BEST MODEL SELECTION
        # =========================
        if score > best_score:
            best_k = k
            best_score = score
            best_centroids = centroids

    print("\n======================")
    print("BEST K:", best_k)
    print("BEST SCORE:", best_score)
    print("======================\n")

    # =========================
    # FINAL TEST PREDICTION
    # =========================
    labels_test = predict_cluster(test, best_centroids)

    df_test = pd.read_csv("data/processed/test.csv")
    df_test["Cluster"] = labels_test
    df_test.to_csv("data/processed/clustered_test.csv", index=False)

    print("Hasil clustering test disimpan ke clustered_test.csv")