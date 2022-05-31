from elementary_algorithm import Evolution
from crossovers import *
from mutations import *
from selections import *


def function_3D(x):
    return x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2


def Rastrigin(x):
    A = 10
    n = 5
    return A * n + np.sum(np.square(x)) - A * np.sum(np.cos(2 * np.pi * x))


evolution = Evolution(crossover_ratio=0.7, mutation_ratio=0.2,
                      crossover=one_point_crossover, mutation=gaussian_mutation,
                      fitness_func=Rastrigin,
                      selection_func=elite_selection, population_size=40,
                      individual_size=5)

evolution.fit(iterations=50)
evolution.visualise()