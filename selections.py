import numpy as np


def elite_selection(scores, n):
    return np.argpartition(scores, -n)[:n]
