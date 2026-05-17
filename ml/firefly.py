# ml/firefly.py
import numpy as np

# ==============================
# HITUNG SSE (fitness)
# ==============================
def calculate_sse(data, centroids):
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    closest_cluster = np.argmin(distances, axis=1)
    
    sse = 0
    for i in range(len(centroids)):
        cluster_points = data[closest_cluster == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[i])**2)
    return sse


# ==============================
# INISIALISASI FIREFLY
# ==============================
def init_fireflies(n_fireflies, k, dim):
    return np.random.rand(n_fireflies, k, dim)


# ==============================
# GERAK FIREFLY
# ==============================
def move_firefly(xi, xj, beta=0.5, gamma=1):
    r = np.linalg.norm(xi - xj)
    attractiveness = beta * np.exp(-gamma * r**2)
    random_step = 0.2 * (np.random.rand(*xi.shape) - 0.5)
    return xi + attractiveness * (xj - xi) + random_step


# ==============================
# MAIN FIREFLY OPTIMIZATION
# ==============================
def run_firefly(data, k, n_fireflies=10, max_iter=20):
    dim = data.shape[1]
    fireflies = init_fireflies(n_fireflies, k, dim)

    for iteration in range(max_iter):
        fitness = np.array([
            calculate_sse(data, f) for f in fireflies
        ])

        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] < fitness[i]:
                    fireflies[i] = move_firefly(fireflies[i], fireflies[j])

    # firefly terbaik = SSE terkecil
    best_idx = np.argmin([
        calculate_sse(data, f) for f in fireflies
    ])

    best_centroids = fireflies[best_idx]
    return best_centroids