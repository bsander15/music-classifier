import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier



# dataset = pd.read_csv('data/music_other_dataset_3sec.csv')
# X = dataset.iloc[:,1:-1].values
# y = dataset.iloc[:,45].values
# X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

# clf = MLPClassifier(hidden_layer_sizes=(44,44,44,44,44), max_iter=500)
# clf.fit(X_train,y_train)
# preds = clf.predict(X_test)
# print(np.mean(y_test == preds))
# print(confusion_matrix(y_test,preds))
# print(classification_report(y_test,preds))
'''
Find best k value
'''

def bestK(X,y):
    all_errors = np.zeros((40,100))
    for x in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

        sample_errors = []
        for i in range(1,41):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train,y_train)
            y_pred = knn.predict(X_test)
            sample_errors.append(np.mean(y_pred != y_test))
        all_errors[:,x] = sample_errors
        # plt.figure(figsize=(10,7))
        # plt.plot(range(1,41),errors, color='red', linestyle='dashed', marker='o')
        # plt.title('Error Rate by K-Value')
        # plt.xlabel('K-Value')
        # plt.ylabel('Error Rate')
        # plt.show()
    k_avg_error = np.mean(all_errors,axis=1)
    k = np.argmin(k_avg_error)
    print(k_avg_error)
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
    def predict(self, data=None):
        if not data:
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
    # data = np.genfromtxt('data/data.csv', delimiter=',')
    model = MOClassifier(KNeighborsClassifier(5))
    print(type(model.classifier))


if __name__ == '__main__':
    main()


