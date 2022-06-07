import numpy as np


def one_point_crossover(parentA, parentB):
    n = parentA.shape[0]
    split_at = n // 2
    child = np.zeros(parentA.shape)

    child[:split_at] = parentA[:split_at]
    child[split_at:] = parentB[split_at:]

    return child


def mean_crossover(parentA, parentB):
    return (parentA + parentB) // 2


def mixed_crossover(parentA, parentB):
    n = parentA.shape[0]
    child = np.zeros(parentA.shape).astype(int)
    A_indices = np.random.randint(0, n, n // 2)
    child[A_indices] = parentA[A_indices]
    mask = np.ones(n, dtype=bool)
    mask[A_indices] = False
    child[mask] = parentB[mask]
    return child

