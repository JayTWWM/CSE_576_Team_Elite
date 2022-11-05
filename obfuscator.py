import argparse
import math
import random
# from classifier.Classifier import Classifier
import pickle
import pandas as pd
from util.Scores import Scores
from util.getNeighbours import GetNeighbours
import os
import gensim.models.keyedvectors as w2v
from pathlib import Path

# import testing

# def getRunandDocumentNumber(indexNo, datasetName, authors):

def getNeighbours(text, neighbours):
    
    model = w2v.KeyedVectors.load_word2vec_format("Word2vec.bin", binary=True)


if __name__ == '__main__':

    mutantPar = {}      # MUTANT-X Parameters
    mutantPar['generation'] = 5      # total number of document generated per document\
    mutantPar['topK'] = 5            # Top K nearest neighbour
    mutantPar['CrossoverProbability'] = 0.1  # Crossover Probability
    mutantPar['iteration'] = 25      # Total Number of iteration
    mutantPar['alpha'] = 0.75        # weight assigned to probability in fitness function
    mutantPar['beta'] = 0.25         # weight assigned to METEOR in fitness function
    mutantPar['changePercentage'] = 0.05     # Percentage of document to change
    mutantPar['replacementList'] = 0.20      # Replacements List

    obfuscatorPar = {}  # Obfuscator Parameters
    obfuscatorPar['author'] = 5     # Total number of authors to keep
    obfuscatorPar['dataset'] = 'amt'    # name of dataset to test with
    obfuscatorPar['neighbours'] = 5     # Total allowed neighbours in word embedding
    obfuscatorPar['runs'] = 1           # Number of runs
    obfuscatorPar['classifier'] = 'ml'  # Classifier

    classifier = obfuscatorPar['classifier']
    nAuthor = obfuscatorPar['author']
    dataset = obfuscatorPar['dataset']
    runs = obfuscatorPar['runs']
    nNeighbours = obfuscatorPar['neighbours']
    fileName = 'avengers_reviews'

    # clf = Classifier(classifier, nAuthor, dataset, runs)
    # clf.loadClassifier()

    # testDataPath = Path("testing_X.csv").resolve()
    data_path = "dataset/test_X.csv"
    testData = pd.read_csv(data_path)

    authorID = testData['author'][0]
    ubText = testData['review'][0]

    scores = Scores(ubText)    
    neighbours = GetNeighbours(scores.words, nNeighbours)


    print("Looking for Word space Dictionary...")
    wordSpacePath = 'wordSpace/'+ fileName + '.pickle' 
    if os.path.isfile(wordSpacePath):
        print('Word Space Dictionary Found.')
        with open(wordSpacePath, 'rb') as w:
            nDict = pickle.load(w)      # loading neighbours dictionary
    else:
        print("Word Space Dictionary not Found, creating one.")
        # neighboursDict = getNeighbours(hText.words, nNeighbours)
        n = neighbours.getNeighbourDict()
        print(n)


