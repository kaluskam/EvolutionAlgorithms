import numpy as np


def gaussian_mutation(chromosome):
    return np.random.normal(0, 1, chromosome.shape[0])


def int_mutation(chromosome):

    vector_to_add = np.random.randint(0, 10, chromosome.shape[0])
    return vector_to_add + chromosome