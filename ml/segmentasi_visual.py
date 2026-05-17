import pandas as pd

def get_cluster_visual_data():
    df = pd.read_csv("data/processed/clustered_train.csv")

    return {
        "recency": df["Recency"].tolist(),
        "frequency": df["Frequency"].tolist(),
        "monetary": df["Monetary"].tolist(),
        "cluster": df["Cluster"].tolist()
    }