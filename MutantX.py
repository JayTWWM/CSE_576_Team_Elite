from meteor.meteor.meteor import Meteor
import random
import copy
import numpy as np
import math
from util.Scores import Scores
import spacy
nlp = spacy.load('en_core_web_sm')

def getPOStag(word):

    parsed = nlp(word)
    for token in parsed:
        return token.pos_

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

    def Fitness(self, originalText, updatedText, classes):

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
        # print(upTextAuthor)
        # ind = np.where(classes == upTextAuthor)[0][0]
        ind = classes.index(upTextAuthor)
        # print(ind)
        updatedTextProb = updatedText.docAuthorProb[ind]
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
        
    
    def single_point_crossover(self, parentDocument1, parentDocument2):

        randomPosition = random.randint(0, len(parentDocument1.inTextTrailingSpaceWords.keys()) - 1)

        crossedOverDocumentText1 = ''
        for i in range(randomPosition - 1):
            (reqWord, _) = parentDocument1.inTextTrailingSpaceWords[i]
            crossedOverDocumentText1 += reqWord
        for i in range(randomPosition, len(parentDocument2.inTextTrailingSpaceWords)):
            (reqWord, _) = parentDocument2.inTextTrailingSpaceWords[i]
            crossedOverDocumentText1 += reqWord

        crossedOverDocumentText2 = ''
        for i in range(randomPosition - 1):
            (reqWord, _) = parentDocument2.inTextTrailingSpaceWords[i]
            crossedOverDocumentText2 += reqWord
        for i in range(randomPosition, len(parentDocument1.inTextTrailingSpaceWords)):
            (reqWord, _) = parentDocument1.inTextTrailingSpaceWords[i]
            crossedOverDocumentText2 += reqWord

        documentDictionary1= parentDocument1.wordReplacementsDict
        documentDictionary1 = {key: value for key, value in documentDictionary1.items() if key < randomPosition}
        documentDictionary2 = parentDocument2.wordReplacementsDict
        documentDictionary2 = {key: value for key, value in documentDictionary2.items() if key >= randomPosition}
        crossedOverDocumentDictionary1 = {**documentDictionary1, **documentDictionary2}

        documentDictionary2 = parentDocument2.wordReplacementsDict
        documentDictionary2 = {key: value for key, value in documentDictionary2.items() if key < randomPosition}
        documentDictionary1 = parentDocument1.wordReplacementsDict
        documentDictionary1 = {key: value for key, value in documentDictionary1.items() if key >= randomPosition}
        crossedOverDocumentDictionary2 = {**documentDictionary1, **documentDictionary2}

        crossedOverDocument1 = Scores(crossedOverDocumentText1)
        crossedOverDocument1.wordReplacementsDict.update(crossedOverDocumentDictionary1)

        crossedOverDocument2 = Scores(crossedOverDocumentText2)
        crossedOverDocument2.wordReplacementsDict.update(crossedOverDocumentDictionary2)

        return crossedOverDocument1, crossedOverDocument2

    def makeReplacement(self, currentDocumentOrig):

        currentDocument = copy.deepcopy(currentDocumentOrig)

        modifiedDocumentText = currentDocument.inTextTrailingSpaceWords
        modifiedDocumentWordReplacementsDict = currentDocument.wordReplacementsDict
        modifiedDocumentAvailableReplacements = currentDocument.availableReplacement

        replacementsCount = math.floor(self.replacements * len(currentDocument.inTextTrailingSpaceWords))

        # Skipping for the mutant
        # if len(modifiedDocumentWordReplacementsDict) > self.replacementsLimit * len(modifiedDocumentText):
        #     return currentDocumentOrig

        i = 0
        while i < replacementsCount:

            # If the available replacements dictionary is empty then stop the process for this mutant
            # and return the parent
            try:
                randomPosition = random.choice(modifiedDocumentAvailableReplacements)
            except:
                return currentDocumentOrig

            # To make sure that a word chose once is never chosen again
            try:
                (randomWord,posTag) = currentDocumentOrig.inTextTrailingSpaceWords[randomPosition]
            except Exception as e:
                modifiedDocumentAvailableReplacements.remove(randomPosition)
                continue

            if posTag not in self.POStags:
                modifiedDocumentAvailableReplacements.remove(randomPosition)
                continue

            if randomWord[-1] == ' ':
                tempRandomWord = randomWord[:-1]
                try:
                    replacement = random.choice(self.sentimentNeighbours[tempRandomWord])
                except:
                    modifiedDocumentAvailableReplacements.remove(randomPosition)
                    continue

                replacementPOSTag = getPOStag(replacement)
                replacement = replacement + ' '

            else:
                tempRandomWord = randomWord
                try:
                    replacement = random.choice(self.sentimentNeighbours[tempRandomWord])
                except:
                    modifiedDocumentAvailableReplacements.remove(randomPosition)
                    continue

                replacementPOSTag = getPOStag(replacement)



            modifiedDocumentText[randomPosition] = (replacement, replacementPOSTag)


            try:
                modifiedDocumentWordReplacementsDict[randomPosition].append((randomWord,replacement))
                # print(modifiedDocumentWordReplacementsDict[randomPosition])
            except KeyError:
                modifiedDocumentWordReplacementsDict[randomPosition] = [(randomWord, replacement)]

            i+=1

        updatedText = ''
        for i in range(len(modifiedDocumentText)):
            (reqWord,_) = modifiedDocumentText[i]
            updatedText+=reqWord

        modifiedDocument = Scores(updatedText)
        modifiedDocument.wordReplacementsDict.update(modifiedDocumentWordReplacementsDict)
        modifiedDocument.availableReplacements = modifiedDocumentAvailableReplacements
        return modifiedDocument
