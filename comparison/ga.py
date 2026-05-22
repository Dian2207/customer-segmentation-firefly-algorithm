import numpy as np
np.random.seed(42)

def calculate_sse(data, centroids):

    distances = np.linalg.norm(
        data[:, None] - centroids,
        axis=2
    )

    closest_cluster = np.argmin(
        distances,
        axis=1
    )

    sse = 0

    for i in range(len(centroids)):

        cluster_points = data[
            closest_cluster == i
        ]

        if len(cluster_points) > 0:

            sse += np.sum(
                (cluster_points - centroids[i]) ** 2
            )

    return sse


def init_population(
    population_size,
    k,
    dim
):

    return np.random.rand(
        population_size,
        k,
        dim
    )

def select_parents(population, fitness):

    idx = np.argsort(fitness)

    parent1 = population[idx[0]]
    parent2 = population[idx[1]]

    return parent1, parent2

def crossover(parent1, parent2):

    alpha = np.random.rand()

    child = (
        alpha * parent1
        +
        (1 - alpha) * parent2
    )

    return child


def mutate(
    child,
    mutation_rate=0.1
):

    mutation = (
        mutation_rate *
        (np.random.rand(*child.shape) - 0.5)
    )

    child = child + mutation

    return np.clip(child, 0, 1)

def run_ga(
    data,
    k,
    population_size=10,
    generations=20
):

    dim = data.shape[1]

    population = init_population(
        population_size,
        k,
        dim
    )

    best_solution = None
    best_fitness = float("inf")

    for generation in range(generations):

        fitness = np.array([

            calculate_sse(data, individual)

            for individual in population

        ])

        idx_best = np.argmin(fitness)

        if fitness[idx_best] < best_fitness:

            best_fitness = fitness[idx_best]

            best_solution = population[idx_best].copy()

        new_population = []

        for _ in range(population_size):

            parent1, parent2 = select_parents(
                population,
                fitness
            )

            child = crossover(
                parent1,
                parent2
            )

            child = mutate(child)

            new_population.append(child)

        population = np.array(new_population)

    return best_solution