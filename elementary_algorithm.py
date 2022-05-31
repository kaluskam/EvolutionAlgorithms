import numpy as np


class Evolution:
    def __init__(self, crossover_ratio, mutation_ratio, crossover, mutation,
                 fitness_func, selection, population_size, individual_size):
        self.crossover_ratio = crossover_ratio
        self.mutation_ratio = mutation_ratio
        self.crossover = crossover
        self.mutation = mutation
        self.fitness_func = fitness_func
        self.selection = selection
        self.population_size = population_size
        self.individual_size = individual_size
        self.population = None
        self.initialize_population()

    def initialize_population(self):
        self.population = np.random.uniform(size=(self.population_size,
                                                  self.individual_size))

    def fit(self):
        stop_condition = True
        while not stop_condition:
            temp_population = self.population
            if np.random.uniform(size=1) <= self.crossover_ratio:
                parentA, parentB = np.random.choice(self.population, size=2)
                child = self.crossover(parentA, parentB)
                temp_population = np.concatenate((self.population, child), axis=1)

            if np.random.uniform(size=1) <= self.mutation_ratio:
                pass

    def select_best_individuals(self):
        fitness_scores = np.zeros(self.temp_population.shape[0])
        for i, individual in enumerate(self.temp_population):
            fitness_scores[i] = self.fitness_func(individual)

