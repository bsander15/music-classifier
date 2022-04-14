import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('music_other_dataset.csv')
# print(dataset.head())

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,45].values

all_errors = np.zeros((40,100))

for x in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    # classifier = KNeighborsClassifier(n_neighbors=5)
    # classifier.fit(X_train,y_train)

    # y_pred = classifier.predict(X_test)
    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))
    # print(y_test,y_pred)
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
