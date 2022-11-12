import pandas as pd
import FeaturesComputation as FC
import numpy as np
from tqdm import tqdm
import pickle

# setting up the parameters
dataset_name = ""
totalAuthors = 0

def column_selector():
    return [i for i in range(0, 546)]

def getData():
    
    # path = "../dataset/"
    # data = pd.read_csv(path+'/train.csv')
    # shuffled = data.sample(frac=1).reset_index()

    # split_l = int(0.8 * len(shuffled))

    # test_X = []
    # test_y = []
    # train_X = []
    # train_y = []
    # print("Getting Training and Testing data...")
    
    # # for i in tqdm(range(0, 100)):
    # #     # print(i)
    # for i in tqdm(range(split_l, len(shuffled))):
    #     text_features = FC.getFeatures(shuffled['review'][i])
    #     test_X.append(text_features)
    #     test_y.append(shuffled['author'][i])
    # for i in tqdm(range(0, split_l)):
    # # for i in tqdm(range(0, 500)):
    #     # print(i)
    #     text_features = FC.getFeatures(shuffled['review'][i])
    #     train_X.append(text_features)
    #     train_y.append(shuffled['author'][i])
    path = "trainedModels/featureSet"
    train_X = pickle.load(open(path+'/X_train.pickle', 'rb'))
    train_y = pickle.load(open(path+'/y_train.pickle', 'rb'))
    test_X = pickle.load(open(path+'/X_test.pickle', 'rb'))
    test_y = pickle.load(open(path+'/y_test.pickle', 'rb'))
    
    return np.matrix(train_X),  train_y, np.matrix(test_X), test_y
        



