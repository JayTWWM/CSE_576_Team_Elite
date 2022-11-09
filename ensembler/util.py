import pandas as pd
import FeaturesComputation as FC
import numpy as np

# setting up the parameters
dataset_name = ""
totalAuthors = 0

def column_selector():
    return [i for i in range(0, 546)]

def getData():
    
    path = "../dataset/"
    test_data = pd.read_csv(path + "test_X.csv")
    train_data = pd.read_csv(path + "train_X.csv")
    print(len(test_data))
    test_X = []
    test_y = []
    train_X = []
    train_y = []
    print("Getting Training and Testing data...")
    for i in range(0, 5):
        print(i)
        text_features = FC.getFeatures(test_data['review'][i])
        test_X.append(text_features)
        test_y.append(test_data['author'][i])
    for i in range(0, 5):
        print(i)
        text_features = FC.getFeatures(train_data['review'][i])
        train_X.append(text_features)
        train_y.append(train_data['author'][i])
    
    return np.matrix(train_X), train_y, np.matrix(test_X), test_y
        



