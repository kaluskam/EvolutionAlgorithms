import numpy as np


def gaussian_mutation(chromosome):
    return np.random.normal(0, 1, chromosome.shape[0])