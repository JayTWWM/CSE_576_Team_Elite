"""
This file contains the code to get topK neighbours using word2vec.

"""
import gensim.models.keyedvectors as w2v
import time


class GetNeighbours:

    def __init__(self, data, nCount):
        self.data = data
        self.totalNeighbours = nCount
        modelPath = "util/word2vec/Word2vec.bin"
        self.word2Vector = w2v.KeyedVectors.load_word2vec_format(modelPath, binary=True)
        self.nDict = {}
        
    
    def getNeighbourDict(self):
        for word in self.data:
            word = str(word).lower()
            flag = True
            try:
                neighbours = list(self.word2Vector.similar_by_word(word, topn=self.totalNeighbours))
            except:
                flag = False        
            upNeighbours = [n[0] for n in neighbours if n[1]>0.75]           
            if flag:
                self.nDict[word]=upNeighbours
        return self.nDict
