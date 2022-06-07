import numpy as np


def gaussian_mutation(chromosome):
    return np.random.normal(0, 1, chromosome.shape[0])


def int_mutation(chromosome):

    vector_to_add = np.random.randint(-5, 5, chromosome.shape[0])
    mutated = vector_to_add + chromosome
    return np.where(mutated < 0, 0, mutated)