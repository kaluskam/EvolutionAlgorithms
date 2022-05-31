import os
import pandas as pd
import numpy as np

from evolution import Evolution
from crossovers import *
from mutations import *
from selections import *


class Rectangle:
    def __init__(self, height, width, value):
        self.height = height
        self.width = width
        self.value = value

    def __repr__(self):
        return f'Rectangle with height: {self.height}, width: {self.width}, ' + \
               f'value: {self.value}'


class Data:
    def __init__(self, radius, rectangles):
        self.radius = radius
        self.rectangles = rectangles
        self.n_rectangles = len(rectangles)

    def fitness_func(self, x):
        values = np.array([rect.value for rect in self.rectangles])
        return -np.sum(np.multiply(x, values))


class DataLoader:
    def __init__(self, dir='cutting'):
        self.dir = dir

    def load(self, filename):
        df = pd.read_csv(os.path.join(self.dir, filename), header=None)
        radius = int(filename.replace('r', '').replace('.csv', ''))
        rectangles = []
        for i, row in df.iterrows():
            rectangles.append(Rectangle(row[0], row[1], row[2]))
        return Data(radius=radius, rectangles=rectangles)


# TODO
def check_constraints():
    # sprwdź czy wszystkie prostokąty się mieszczą
    # pomysł - algorytm zachłanny
    pass
r800 = DataLoader().load('r800.csv')
e800 = Evolution(crossover_ratio=0.7, mutation_ratio=0.2,
                 crossover=mean_crossover, mutation=int_mutation,
                 fitness_func=r800.fitness_func,
                 selection_func=elite_selection, population_size=40,
                 individual_size=r800.n_rectangles)

e800.fit(iterations=100)

