import numpy as np

np.random.seed(None)


def calculate_sse(data, centroids):
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    sse = 0

    for i in range(len(centroids)):
        cluster_points = data[labels == i]

        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[i]) ** 2)

    return sse


def init_fireflies(data, n_fireflies, k, init_centroids=None):

    fireflies = []

    for i in range(n_fireflies):

        # Firefly pertama menggunakan centroid KMeans
        if i == 0 and init_centroids is not None:
            fireflies.append(init_centroids.copy())

        else:
            idx = np.random.choice(len(data), k, replace=False)
            fireflies.append(data[idx])

    return np.array(fireflies)


def move_firefly(xi, xj, data, beta=1.0, gamma=0.3, alpha=0.02):

    r = np.linalg.norm(xi - xj)

    attractiveness = beta * np.exp(-gamma * r**2)

    random_step = alpha * (np.random.rand(*xi.shape) - 0.5)

    new_x = xi + attractiveness * (xj - xi) + random_step

    # Boundary control
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    new_x = np.clip(new_x, data_min, data_max)

    return new_x


def run_firefly(
    data,
    k,
    init_centroids=None,
    n_fireflies=10,
    max_iter=30
):

    fireflies = init_fireflies(
        data,
        n_fireflies,
        k,
        init_centroids
    )

    best_firefly = None
    best_fitness = float("inf")

    for iteration in range(max_iter):

        fitness = np.array([
            calculate_sse(data, f)
            for f in fireflies
        ])

        for i in range(n_fireflies):

            for j in range(n_fireflies):

                if fitness[j] < fitness[i]:

                    fireflies[i] = move_firefly(
                        fireflies[i],
                        fireflies[j],
                        data
                    )

        fitness = np.array([
            calculate_sse(data, f)
            for f in fireflies
        ])

        best_idx = np.argmin(fitness)

        if fitness[best_idx] < best_fitness:

            best_fitness = fitness[best_idx]
            best_firefly = fireflies[best_idx].copy()

    return best_firefly