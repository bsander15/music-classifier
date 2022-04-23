from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/Volumes/Personal/jhimel22/CS5100/Classifications/features_30_sec.csv')

# to normalize the dataset
scaling = StandardScaler()
#X = scaling.fit_transform(np.array(data.ioc[:, :-1]))

# fit and scale data
scaling.fit(data.drop(labels="filename", axis = 1))
scaled = scaling.transform(data.drop(labels="filename", axis = 1))

scaled_df = pd.DataFrame(scaled, columns = data.columns[:-1])
print(scaled_df.head())

# get genre names (last column in csv)
genres = data.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(genres)
X = scaled_df


# # input X is music features, output Y is the target genre
# X = data_features
# y = data_features['target']

# random shuffle to ensure split is randomized and therefore representative of entire data
# 70% data trained
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=30, shuffle=True)

# K = 7
knn = KNeighborsClassifier(n_neighbors=7, p=2, metric = 'euclidean')
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
prediction()
print("Classification Report: ", classification_report(y_test, prediction))

training_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)
print("Training Accuracy: ", training_accuracy)
print("Test Accuracy: ", test_accuracy)

# error charting
error_frequency = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    prediction_i = knn.predict(X_test)
    error_frequency.append(np.mean(prediction_i != y_test))

k = np.argmin(error_frequency)
print(k)

# to determine best K (visual representation of error charting
plt.figure(figsize = (10,5))
plt.plot(range(1,40), error_frequency, color = 'green')
plt.title("Error Frequency Per K Value")
plt.xlabel("K")
plt.ylabel("Error Frequency")

# confusion matrix for false negatives and true positives
confusion_matrix = confusion_matrix(y_test, prediction)
confusion_matrix()

