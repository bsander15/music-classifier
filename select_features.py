
from extract_features import *
import random


class SelectFeatures:
    def __init__(self, signals, model):
        self.signals = signals
        self.model = model
        self.features = ExtractFeatures(self.signals, feature_names)

    def rand_individual(self, size=8):
        return random.sample(self.features, k=size)

    def evaluate(self, feature_names):
        # use extract features
        # run on ml model
        pass

    def optimize(self, population_size=100):
        population = [self.rand_individual() for _ in population_size]

    def mutate(self):
        pass

    def crossover(self):
        pass

""" neural net -- change different number of layers """
