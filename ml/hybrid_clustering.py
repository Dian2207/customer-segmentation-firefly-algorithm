import pandas as pd
import numpy as np
from ml.firefly import run_firefly
from ml.kmeans import run_kmeans

def run_hybrid_clustering(k=4):
    print("STEP 5: TRAIN CLUSTERING")

    train = pd.read_csv("data/processed/train.csv").values

    centroids = run_firefly(train, k)

    if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):
        print("WARNING: centroid invalid, reinitializing...")
        idx = np.random.choice(len(train), k, replace=False)
        centroids = train[idx]

    labels, centroids = run_kmeans(train, centroids)

    df = pd.read_csv("data/processed/train.csv")
    df["Cluster"] = labels
    df.to_csv("data/processed/clustered_train.csv", index=False)
    print("Clustering selesai dan disimpan")

    return centroids