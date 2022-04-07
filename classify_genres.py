
import json
import numpy as np

from sklearn.model_selection import train_test_split
# import tensorflow.keras as keras

# load data
# split data into train and test sets
# build network architecture
# compile network
# train network

"""
    Temporary file for testing the MFCCs with ML models
"""

DATASET_PATH = 'mfccs.json'

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    # convert lists into numpy arrays 
    inputs = np.array(data['mfcc']) 
    targets = np.array(data['labels'])

    return inputs, targets


if __name__ == '__main__':

    # load data
    inputs, targets = load_data(DATASET_PATH)

    inputs_train, inputs_test, targets_train, targets_test = \
        train_test_split(inputs, targets, test_size=0.3)
    
    # model = keras.Sequential([
    #     # input layer
    #     keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

    #     keras.layers.Dense(512, activation='relu')
    # ])