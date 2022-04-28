
from re import A
import pandas as pd
from sklearn.metrics import classification_report
from extract_features import *
from classifiers.MOClassifier import MusicKNN, MusicNet
import random

MFCC_SET = [f'mfcc{i}' for i in range(1, 20)]
FEATURE_SET = ['sc', 'zcr', 'rolloff'] + MFCC_SET
 
class SelectFeatures:
    def __init__(self, signals=None, model=None, length_range=(8, 36)):
        self.signals = signals
        self.model = model
        self.features = [ f'{pref}_{suf}' for pref in {'mean', 'var'} for suf in FEATURE_SET] + ['tempo']
        self.length_range = length_range

    def rand_feature(self):
        num_features = len(self.features)
        return self.features[random.randint(0, num_features-1)]

    def rand_individual(self):
        a,b = self.length_range
        rand_size = random.randint(a, b)
        return random.sample(self.features, k=rand_size)

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

        # note: this sorts by cost in ascending order (by accuracy)
        population.sort(key = lambda g : g[1], reverse=True)

        for _ in range(generations):
            new_population = []            
            print(len(population))
            for i in range(0, len(population), 2):
                a, b  = population[i], population[i+1]
                parent1, parent2 = a[0], b[0]
                child1 = self.reproduce(parent1,parent2, length=len(parent1))
                child2 = self.reproduce(parent1,parent2, length=len(parent2))
                new_population.append( (child1, self.evaluate(child1)) )
                new_population.append( (child2, self.evaluate(child2)) )

            population = sorted(new_population, key = lambda g : g[1], reverse=True)

        return population[0][0] # return subset of features with highest score

    def interleave_uniq(self, parent1, parent2):
        i = 0; j = 0;
        n = len(parent1); m = len(parent2);
        is_i = True
        interleaved = []
        while i < n and j < m:
            if is_i:
                if parent1[i] not in interleaved:
                    interleaved.append(parent1[i])
                i+=1
            else:
                if parent2[j] not in interleaved:
                    interleaved.append(parent2[j])
                j+=1
            is_i = not is_i

        return interleaved

    # takes in as input two parent lists of features (no costs are provided)
    # called on both parents' lengths
    def reproduce(self, parent1, parent2, length=6):

        # crossover
        child = self.interleave_uniq(parent1, parent2) 

        # if child is too small, add random features until it matches the individual size
        while len(child) < length: 
            rand_feature = self.rand_feature()
            if rand_feature not in child:
                child.append(rand_feature)

        # if child is too long, clip the end
        child = child[:length] 

        # mutate: randomly choose a gene to randomly change
        rand_gene_pos = random.randint(0, length-1)

        # keep trying until that feature is unique in the feature vector
        rand_feature = self.rand_feature()
        while rand_feature in child:
            rand_feature = self.rand_feature()
        child[rand_gene_pos] = rand_feature

        return child


""" neural net -- change different number of layers """

if __name__ == '__main__':
    sf = SelectFeatures()
    # ind = sf.rand_individual()
    # print(ind, len(ind))