from meteor.meteor.meteor import Meteor
import random

class MutantX:

    def __init__(self, generation, topK, crossover, iterations, sentimentNeighbours, alpha, beta, replacements, rLimit):

        self.generation = generation
        self.topK = topK
        self.crossover = crossover
        self.iterations = iterations
        self.sentimentNeighbours = sentimentNeighbours
        self.alpha = alpha
        self.beta = beta
        self.replacements = replacements
        self.rLimit = rLimit
        self.alpha = alpha
        self.beta = beta

        self.POStags = ['ADJ', 'ADP', ' ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'PRON', 'SCONJ', 'VERB']

        self.meteor = Meteor()

    def Fitness(self, originalText, updatedText):

        # checking the total length of text 
        if len(originalText.documentText.split()) < 3000:
            ut_ = (''.join(str(e) for e in updatedText.documentText.split()))
            ot_ = [''.join(str(e) for e in originalText.documentText.split())]
            meteorScore = self.meteor._score(ut_, ot_)
        else:
            ut_ = (''.join(str(e) for e in updatedText.documentText.split()[:3000]))
            ot_ = [''.join(str(e) for e in originalText.documentText.split()[:3000])]
            meteorScore = self.meteor._score(ut_, ot_)
        

        upTextAuthor = updatedText.docAuthor
        updatedTextProb = updatedText.docAuthorProb[upTextAuthor]
        totalReplace = len(updatedText.wordReplacementsDict)

        # calculating fitness score of updated text 
        fitness = (self.alpha * updatedTextProb) + (self.beta * (1 - meteorScore))
        fitness = 1/fitness
        
        # updating score in Score class
        updatedText.meteor = meteorScore
        updatedText.fitness = fitness
        updatedText.clfProb = updatedTextProb
        updatedText.totalReplace = totalReplace

        return meteorScore, fitness, updatedTextProb, totalReplace, upTextAuthor
        


