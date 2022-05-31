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