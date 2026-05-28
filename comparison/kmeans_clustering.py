import pandas as pd
import numpy as np

from ml.kmeans import run_kmeans
from ml.kmeans import elbow_method


def find_elbow_k(k_values, wcss_values):

    points = np.column_stack((k_values, wcss_values))

    first_point = points[0]
    last_point = points[-1]

    line_vector = last_point - first_point
    line_vector = line_vector / np.linalg.norm(line_vector)

    distances = []

    for point in points:

        vector = point - first_point

        projection = np.dot(vector, line_vector) * line_vector

        perpendicular = vector - projection

        distances.append(np.linalg.norm(perpendicular))

    best_index = np.argmax(distances)

    return k_values[best_index]


def run_kmeans_clustering():

    print("\nSTEP 5: TRAIN CLUSTERING K-MEANS")

    df = pd.read_csv(
        "data/processed/train.csv"
    )

    data = df.values

    print("Jumlah data:", data.shape)

    print("\nMencari K terbaik...")

    k_values, wcss_values = elbow_method(data)

    best_k = find_elbow_k(
        k_values,
        wcss_values
    )

    print("K terbaik:", best_k)

    np.random.seed(None)

    init_centroids = data[
        np.random.choice(
            len(data),
            best_k,
            replace=False
        )
    ]

    labels, centroids = run_kmeans(
        data,
        init_centroids
    )

    df["Cluster"] = labels

    df.to_csv(
        "data/result/clustered_train_kmeans.csv",
        index=False
    )

    print(
        "\nFile tersimpan: "
        "data/result/clustered_train_kmeans.csv"
    )

    return labels, centroids, best_k