
from extract_features import *
import random

 
class SelectFeatures:
    def __init__(self, signals, model):
        self.signals = signals
        self.model = model
        self.features = { f'{pref}_{suf}' for pref in {'mean', 'var'} for suf in {'sc', 'mfcc', 'rolloff', 'beat'}}

    def rand_individual(self, size=8):
        return random.sample(self.features, k=size)

    def evaluate(self, feature_names):
        # use extract features
        # run on ml model
        pass

    def optimize(self, population_size=100):
        population = [self.rand_individual() for _ in population_size]

    def reproduce(self, parent1, parent2):
        pass


""" neural net -- change different number of layers """
