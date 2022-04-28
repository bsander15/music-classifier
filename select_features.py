
from re import A
import pandas as pd
from sklearn.metrics import classification_report
from extract_features import *
from classifiers.MOClassifier import MusicKNN, MusicNet
import random

 
class SelectFeatures:
    def __init__(self, signals, model, individual_size=6):
        self.signals = signals
        self.model = model
        self.features = [f'{pref}_{suf}' for pref in {'mean', 'var'} for suf in {'sc', 'mfcc', 'rolloff', 'beat'}]
        self.individual_size = individual_size

    def rand_feature(self):
        num_features = len(self.features)
        return self.features[random.randint(0, num_features-1)]

    def rand_individual(self):
        return random.sample(self.features, k=self.individual_size)

    def evaluate(self, feature_names):
        full_data = pd.read_csv('data/data.csv')
        data = full_data[feature_names]
        if str.lower(self.model) is 'knn':
            agent = MusicKNN(data.values,full_data.iloc[:,-1])
            preds = agent.predict(agent.classifier)
            classification_report = agent.metrics(preds,dict=True)[1]
            return classification_report['accuracy']

        

    def optimize(self, population_size=100, generations=50):

        population = []

        for _ in range(population_size):
            individual = self.rand_individual()
            cost = self.evaluate(individual)
            population.append((individual, cost))

        # note: this sorts by cost in ascending order (may have to be reversed if better accuracy
        # means better score)
        population.sort(key = lambda g : g[1])

        for _ in range(generations):
            new_population = []            
            for i in range(0, len(population)-1):
                a, b  = population[i], population[i+1]
                parent1, parent2 = a[0], b[0]
                child1 = self.reproduce(parent1,parent2)
                child2 = self.reproduce(parent1,parent2)
                new_population.append( (child1, self.evaluate(child1)) )
                new_population.append( (child2, self.evaluate(child2)) )

            population = new_population.sort(key = lambda g : g[1])

        return population[0][0] # return subset of features with highest score


    # takes in as input two parent lists of features (no costs are provided)
    def reproduce(self, parent1, parent2):

        # crossover
        child = set(parent1 + parent2) # combine parents into list of no duplicates
        random.shuffle(child)

        # if child is too small, add random features until it matches the individual size
        while len(child) < self.individual_size: 
            rand_feature = self.random_feature()
            if rand_feature not in child:
                child.append(rand_feature)

        # if child is too long, clip the end
        child = child[:self.individual_size] 

        # mutate: randomly choose a gene to randomly change
        rand_gene_pos = random.randint(0, len(child)-1)

        # keep trying until that feature is unique in the feature vector
        rand_feature = self.rand_feature()
        while rand_feature in child:
            rand_feature = self.rand_feature()
        child[rand_gene_pos] = rand_feature



    

""" neural net -- change different number of layers """
