import pandas as pd
from ml.firefly import run_firefly
from ml.kmeans import run_kmeans

def run_hybrid_clustering(k=4):
    print("STEP 5: TRAIN CLUSTERING")

    train = pd.read_csv("data/processed/train.csv").values

    centroids = run_firefly(train, k)
    labels, centroids = run_kmeans(train, centroids)

    df = pd.read_csv("data/processed/train.csv")
    df["Cluster"] = labels
    df.to_csv("data/processed/clustered_train.csv", index=False)

    return centroids