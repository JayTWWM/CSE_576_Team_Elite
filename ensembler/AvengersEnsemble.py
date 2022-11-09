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

TOTAL_CLFS = 10

def pipeHiddenClf(features):
    # Function to create pipelines 
    classifiers = [make_pipeline(
        ColumnSelector(cols=random.sample(range(features), 30)),
        SVC(kernal='linear', probability=True, random_state=0)
    ) for _ in features]
    return classifiers

def ensemble1(features):
    return make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        EnsembleVoteClassifier(pipeHiddenClf(features), voting='soft', use_clones=False)
    )

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = util.getData()

    columns = ColumnSelector(cols=util.column_selector())  # to select specific columns from the dataset
    variance_thresh = VarianceThreshold()   # Feature selector that remove all low-variance features

    totalFeatuers = len(variance_thresh.fit_transform(columns.fit_transform(X_train))[0])

    ensemble = ensemble1(totalFeatuers)
    ensemble = make_pipeline(columns, variance_thresh, ensemble)

    print("Starting Training")
    ensemble.fit(X_train, y_train)

    if not os.path.exists("trainedModels/amazonData" + "/" ):
        os.makedirs("trainedModels/amazonData" + "/")

    filename = "trainedModels/amazonData" + "/" + f'amazonData.sav'
    pickle.dump(ensemble, open(filename, 'wb'))


    print("accuracy score")
    print(accuracy_score(y_test, ensemble.predict(X_test)))