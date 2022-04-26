import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Volumes/Personal/jhimel22/CS5100/Classifications/features_30_sec.csv')
df = data.drop(labels='filename', axis = 1)

datapath = "/path/to/dataset/in/json/file"

# # to normalize the dataset
# scaling = StandardScaler()
# #X = scaling.fit_transform(np.array(data.ioc[:, :-1]))
#
# # fit and scale data
# scaling.fit(data.drop(labels="filename", axis = 1))
# scaled = scaling.transform(data.drop(labels="filename", axis = 1))
#
# scaled_df = pd.DataFrame(scaled, columns = data.columns[:-1])
# print(scaled_df.head())


def load (datapath):

    with open(datapath, "r") as fp:
        data = json.load(fp)

    X = np.array(data[:,-1])
    y = np.array(data["labels"])

    return X,y

def predict(network, X, y):

    prediction = network.predict(X)

    X = X[np.newaxis,...]

    index_predict = np.argmax(prediction, axis = 1)

    print("Target: {}, Predicted Label: {}".format(y, index_predict))

    #test_predict = network.predict_classes(training)

def evaluate():

    # history = network.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30)

    # evaluate model
    test_loss, accuracy = network.evaluate(X_test, y_test, verbose = 2)
    print(test_loss, accuracy)

    # isolate a sample for prediction
    X_predict = X_test[1]
    y_predict = y_test[1]

    predict(network, X_predict, y_predict)



    # class labeling
classes = []
label_encoding = LabelEncoder()
label_encoding.fit(classes)

num_classes = len(label_encoding.classes_)
print(num_classes, "Classes: " .join(list(label_encoding.classes_)))

n_classes = label_encoding.transform(classes)

if __name__ == "__main__":

    X,y = load(datapath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # network of input later, 1st 2nd and 3rd dense layers, and output layer
    network = keras.Sequential([
        keras.layers.Flatten(input_shape = (X.shape[1], X.shape[2])),
        keras.layers.Dense(512, activation = 'relu'),
        keras.layers.Dense(256, actiation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ])

    # optimize network
    optimize = keras.optimizers.Adam(learning_rate = 0.0001)
    network.compile(optimizer = optimize, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    network.summary()

    # training the network
    training = network.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, eopchs = 50)




#
# predictions = []
# for x in range (test_data):
#     predictions.append(nearestClass(getNeighbors(training_data ,test_data[x] , 5)))
#
# accuracy1 = getAccuracy(test_data , predictions)
# print(accuracy1)
#
# # gets distance between feature vectors, finds neighbors
# def getNeighbors(training_data, instance, k):
#
#     distances = []
#
#     for x in range (len(training_data)):
#         dist = distance(training_data[x], instance, k )+ distance(instance, training_data[x], k)
#         distances.append((training_data[x][2], dist))
#
#     distances.sort(key=operator.itemgetter(1))
#     neighbors = []
#
#     for x in range(k):
#         neighbors.append(distances[x][0])
#     return neighbors
#
# # find nearest neighbors
# def nearestClass(neighbors):
#
#     # class vote
#     vote = {}
#
#     # vote delegation
#     for x in range(len(neighbors)):
#         response = neighbors[x]
#         if response in vote:
#             vote[response]+=1
#         else:
#             vote[response]=1
#
#     sorter = sorted(vote.items(), key = operator.itemgetter(1), reverse=True)
#     return sorter[0][0]
#
# # accuracy evaluation
# def getAccuracy(testSet, predictions):
#     correct = 0
#
#     for x in range (len(testSet)):
#         if testSet[x][-1]==predictions[x]:
#             correct+=1
#
#     return 1.0*correct/len(testSet)
#
#
#
#
#
#
#
# def distance(instance1 , instance2 , k ):
#     distance =0
#     mm1 = instance1[0]
#     cm1 = instance1[1]
#     mm2 = instance2[0]
#     cm2 = instance2[1]
#     distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
#     distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 ))
#     distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
#     distance-= k
#     return distance
#
#
#
# results=defaultdict(int)
#
# i=1
# for folder in os.listdir("./musics/wav_genres/"):
#     results[i]=folder
#     i+=1
#
# (rate,sig)=wav.read("__path_to_new_audio_file_")
# mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
# covariance = np.cov(np.matrix.transpose(mfcc_feat))
# mean_matrix = mfcc_feat.mean(0)
# feature=(mean_matrix,covariance,0)
#
# pred=nearestClass(getNeighbors(dataset ,feature , 5))
#
# print(results[pred])
#

