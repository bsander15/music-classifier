import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
'''
Find best k value
'''

def bestK(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    errors = []
    for i in range(1,41):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        errors.append(np.mean(y_pred != y_test))
    k = np.argmin(errors)
    print(errors)
    print(k)

class MOClassifier:

    """
    Initialize KNN object with default k value of 5 (from testing above)
    and fit classifier using input data
    X: feature matrix
    y: classification labels
    """
    def __init__(self,classifier):
        self.classifier = classifier
    
    def fit(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.2)
        self.classifier.fit(self.X_train, self.y_train)
    """
    Returns classifications of input data
    data: feature matrix (num features must match num features fit with)
    """
    def predict(self, data=np.array([])):
        if data.size == 0:
            data = self.X_test
        return self.classifier.predict(data)
    
    """
    Returns confusion matrix, precision, recall, f1score, and accuracy of predictions
    preds: classification predictions generated
    """
    def metrics(self,preds, labels=None, dict=False):
        if labels == None:
            return confusion_matrix(self.y_test,preds), classification_report(self.y_test,preds,output_dict=dict)
        return confusion_matrix(labels,preds), classification_report(labels,preds,output_dict=dict)
    
    def getClassifier(self):
        return self.classifier
    
    def setClassifier(self, classifier):
        self.classifier = classifier
    
        
def main():
    pass
if __name__ == '__main__':
    main()


