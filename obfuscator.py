import argparse
import math
import random
from classifier.Classifier import Classifier
import pickle
import pandas as pd
from util.Scores import Scores
from util.getNeighbours import GetNeighbours
import os
import copy
import gensim.models.keyedvectors as w2v
from pathlib import Path
from MutantX import MutantX
import numpy as np
from tqdm import tqdm
from pathlib import Path
# import testing

PROJECT_ROOT = Path(__file__).parents[1]


if __name__ == '__main__':

    mutantPar = {}      # MUTANT-X Parameters
    mutantPar['generation'] = 5      # total number of document generated per document\
    mutantPar['topK'] = 5            # Top K nearest neighbour
    mutantPar['CrossoverProbability'] = 0.1  # Crossover Probability
    mutantPar['iteration'] = 25      # Total Number of iteration
    mutantPar['alpha'] = 0.75        # weight assigned to probability in fitness function
    mutantPar['beta'] = 0.25         # weight assigned to METEOR in fitness function
    mutantPar['changePercentage'] = 0.05     # Percentage of document to change
    mutantPar['replacementLimit'] = 0.20      # Replacements List

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
    generation = mutantPar['generation']
    topK = mutantPar['topK']
    c_prob = mutantPar['CrossoverProbability']
    iter = mutantPar['iteration']
    alpha = mutantPar['alpha']
    beta = mutantPar['beta']
    change_p = mutantPar['changePercentage']
    replace_limit = mutantPar['replacementLimit']

    data_path = "dataset/train_X.csv"
    testData = pd.read_csv(data_path)

    authorID = testData['author'][0]
    ubText = testData['review'][0]

    model = Classifier(authorID, runs)
    model.load_Classifier()
    classes = model.callClasses().tolist()

    text_scores = Scores(ubText)    
    neighbours = GetNeighbours(text_scores.words, nNeighbours)

    model.get_label_and_probabilities(text_scores)
    # loading pre trained classifier
    

    print("Looking for Word space Dictionary...")
    wordSpacePath = 'wordSpace/'+ fileName + '.pickle' 

    if os.path.isfile(wordSpacePath):
        print('Word Space Dictionary Found.')
        with open(wordSpacePath, 'rb') as w:
            neighbours_dict = pickle.load(w)      # loading neighbours dictionary
    else:
        print("Word Space Dictionary not Found, creating one.")
        # neighboursDict = getNeighbours(hText.words, nNeighbours)
        neighbours_dict = neighbours.getNeighbourDict()
        if not os.path.exists('wordSpace'):
            os.makedirs('wordSpace')
        with open(wordSpacePath, 'wb') as writer:
            pickle.dump(neighbours_dict, writer)

    # Mutant-X
    mutantX = MutantX(generation, topK, c_prob, iter, neighbours_dict, alpha, beta, change_p, replace_limit)

    # Obfuscation
    allDocs = [text_scores]
    iterCount = 1

    for i in tqdm(range(1, mutantX.iterations)):

        generatedDocsList = []
        # generating documents using mutantX
        for doc in allDocs:
            for i in range(0, mutantX.generation):

                pCopy = copy.deepcopy(doc)
                generatedDoc = mutantX.makeReplacement(doc)
                model.get_label_and_probabilities(generatedDoc)
                generatedDocsList.append(generatedDoc)

        allDocs.extend(generatedDocsList)

        # Crossover
        if random.random() < mutantX.crossover:
            print("In crossover process")
            
            child1, child2 = random.sample(allDocs, 2)
            child1Copy = copy.deepcopy(child1)
            child2Copy = copy.deepcopy(child2)

            crossover_child1, crossover_child2 = mutantX.single_point_crossover(child1Copy, child2Copy)

            model.get_label_and_probabilities(crossover_child1)
            model.get_label_and_probabilities(crossover_child2)

            allDocs.extend([crossover_child1, crossover_child2])
        
        if text_scores in allDocs:
            allDocs.remove(text_scores)
        
        # calculating fitness

        for doc in allDocs:

            meteor, fitnessScore, probability, totalReplace, changedDocAuthor = mutantX.Fitness(text_scores, doc, classes)
            
            print("METEOR SCORE-", meteor)
            print("Fitness Score-", fitnessScore)
            print("probability-", probability)
            print("Total Replacements-", totalReplace)
            print("Changed Document Author-", changedDocAuthor)

            if (text_scores.docAuthor != doc.docAuthor):
                print("Obfuscated successfully!!!", doc.docAuthor)

        # selecting topK
        allDocs.sort(key=lambda x:x.fitness, reverse=True)
        allDocs = allDocs[:mutantX.topK]

        bestIndivisual = allDocs[0]


        for topDocument in allDocs:
            # print("author-",originalDocument.documentAuthor)
            ind = classes.index(text_scores.docAuthor)
            print('CURRENT PROBABILITY : ',
                  topDocument.docAuthorProb[ind])




