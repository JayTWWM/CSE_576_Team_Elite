'''
This file calls the ensemble architecture and save the trained model.
'''

import FeaturesComputation as FC
import util
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector
import random
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import pickle
import numpy as np

TOTAL_CLFS = 10

def pipeHiddenClf(features):
    # Function to create pipelines 
    classifiers = [make_pipeline(
        ColumnSelector(cols=random.sample(range(features), 30)),
        SVC(kernel='linear', probability=True, random_state=0),
    ) for _ in range(TOTAL_CLFS)]
    return classifiers

def ensemble1(features):
    return make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        EnsembleVoteClassifier(pipeHiddenClf(features), voting='soft', use_clones=False)
    )

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = util.getData()

    columns = ColumnSelector(cols=util.column_selector())  # to select specific columns from the dataset
    variance_thresh = VarianceThreshold()   # Feature selector that remove all low-variance features

    totalFeatuers = len(variance_thresh.fit_transform(columns.fit_transform(X_train))[0])

    ensemble = ensemble1(totalFeatuers)
    ensemble = make_pipeline(columns, variance_thresh, ensemble)
    print(y_train)

    print(np.unique(y_train))

    print("Starting Training")
    ensemble.fit(X_train, y_train)
    
    if not os.path.exists("../trainedModels/amazonData" + "/" ):
        os.makedirs("../trainedModels/amazonData" + "/")

    filename = "../trainedModels/amazonData" + "/" + 'amazonData.sav'
    pickle.dump(ensemble, open(filename, 'wb'))

    print("Accuracy Score")
    print(accuracy_score(y_test, ensemble.predict(X_test)))

    if not os.path.exists("../trainedModels/featureSet" + "/" ):
        os.makedirs("../trainedModels/featureSet" + "/")

    pickle.dump(X_train, open('../trainedModels/featureSet/X_train.pickle', 'wb'))
    pickle.dump(y_train, open('../trainedModels/featureSet/y_train.pickle', 'wb'))
    pickle.dump(X_test, open('../trainedModels/featureSet/X_test.pickle', 'wb'))
    pickle.dump(y_test, open('../trainedModels/featureSet/y_test.pickle', 'wb'))