import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Evolution:
    def __init__(self, crossover_ratio, mutation_ratio, crossover, mutation,
                 fitness_func, selection_func, population_size,
                 individual_size):
        self.history = {'mean_score': [], 'best_score': [], 'iteration': []}
        self.new_population = None
        self.parents = None
        self.crossover_ratio = crossover_ratio
        self.mutation_ratio = mutation_ratio
        self.crossover = crossover
        self.mutation = mutation
        self.fitness_func = fitness_func
        self.selection_func = selection_func

        assert population_size % 2 == 0
        self.population_size = population_size

        self.individual_size = individual_size
        self.population = None
        self.initialize_population()
        self.name = f'{crossover.__name__}&{mutation.__name__}&population={population_size}'

    def initialize_population(self):
        self.population = np.random.randint(low=0, high=15, size=(self.population_size,
                                                  self.individual_size))

    def fit(self, iterations=20):
        self.name += f'&iterations={iterations}'
        t = 0
        while t < iterations:
            self.new_population = list(self.population)
            self.match_parents()
            for i in range(0, self.population_size, 2):
                if np.random.uniform(size=1) <= self.crossover_ratio:
                    parentA, parentB = self.parents[i], self.parents[i + 1]
                    child = self.crossover(parentA, parentB)
                    self.new_population.append(child)

            for i in range(len(self.new_population)):
                if np.random.uniform(size=1) <= self.mutation_ratio:
                    self.new_population[i] = self.mutation(
                        self.new_population[i])
            fitness_scores = self.evaluate()
            self.print_results(fitness_scores, t)
            self.selection(fitness_scores)
            self.print_evaluate_new_generation()
            self.save_to_history(t)
            t += 1
        self.best_individual = self.new_population[np.argmax(fitness_scores)]

    def save_to_history(self, iteration):
        scores = np.apply_along_axis(self.fitness_func, 1, self.population)
        self.history['mean_score'].append(np.mean(scores))
        self.history['best_score'].append(np.max(scores))
        self.history['iteration'].append(iteration)

    def save_history_to_csv(self):
        df = pd.DataFrame(self.history)
        df.to_csv(f'history\\{self.name}.csv')

    def visualise(self):
        plt.figure(figsize=[10, 6])
        plt.subplot(1, 2, 1)
        plt.plot(self.history['iteration'], self.history['mean_score'])
        plt.title('Mean score of population over time')
        plt.subplot(1, 2, 2)
        plt.plot(self.history['iteration'], self.history['best_score'])
        plt.title('Best score from population over time')
        plt.savefig(f'plots\\{self.name}.png')

    def print_results(self, scores, generation_num):
        print(f'Generation {generation_num} best score is {np.max(scores)} for individual '
              f'{self.new_population[np.argmax(scores)]}')

    def print_evaluate_new_generation(self):
        scores = np.apply_along_axis(self.fitness_func, 1, self.population)
        print(f'Population average score is {np.mean(scores)}')

    def match_parents(self):
        self.parents = copy.deepcopy(self.population)
        np.random.shuffle(self.parents)

    # def select_best_individuals(self):
    #     fitness_scores = np.zeros(self.temp_population.shape[0])
    #     for i, individual in enumerate(self.temp_population):
    #         fitness_scores[i] = self.fitness_func(individual)

    def evaluate(self):
        x = np.array(self.new_population)
        return np.apply_along_axis(self.fitness_func, 1, x)

    def selection(self, scores):
        indices = self.selection_func(scores, self.population_size)
        self.population = np.array(self.new_population)[indices.astype('int')]


