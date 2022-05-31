import numpy as np


def selection(optimum, individual):
    return np.linalg.norm(optimum - individual)