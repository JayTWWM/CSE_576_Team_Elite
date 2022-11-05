import spacy
# en_core_web_sm is optimized pipeline for CPU in english language.  
nlp = spacy.load('en_core_web_sm')

class Scores:

    def __init__(self, data):

        self.documentText = ''      # contains text withour space character
        self.words = []             # contains words in text
        self.docAuthor = 0          # author of document
        self.docAuthorProb = 0      # Probability of document author

        self.availableReplacement = []  

        self.meteor = 0.0       # METEOR score of the text
        self.fitness = 0.0      # fitness score of the text
        self.clfProb = 0.0      
        self.totalReplace = 0.0

        data_ = nlp(str(data))
        self.inTextTrailingSpaceWords = {}

        index = 0

        for d in data_:

            self.words.append(str(d.text))
            self.inTextTrailingSpaceWords[index] = (str(d.text_with_ws), str(d.pos_))
            self.documentText+=str(d.text_with_ws)
            self.availableReplacement.append(index)
            index+=1
        self.wordReplacementsDict = {}