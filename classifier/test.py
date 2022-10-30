from cgi import test
import os 
import pickle 
import re
import nltk
from keras_preprocessing import text

name = 'amt'
no = 5


def getFeatures(data):
    input = text.text_to_word_sequence(data, filters=",.?!\"'`;:-()&$", lower=True, split=" ")

    inputText = str(data).lower()
    in_p = inputText.lower().replace(" ", "")

    p_path = 'C:/Users/sshar244/Desktop/codes/avengers_ensemble/classifier/punctuation.txt'
    p_words = open(p_path, "r").readlines()
    p_words = [i.strip("\n") for i in p_words]

    count = []
    for i in range(0, len(p_words)):
        count.append(in_p.count(p_words[i]))
    
    print(count)








if(os.path.exists(f'datasets/{name}-{no}')):
    X_test = pickle.load(open(f'C:/Users/sshar244/Desktop/codes/avengers_ensemble/datasets/{name}-{no}/X_test.pickle', 'rb'))
else:
    print("No")

for (filePath, filename, authorId, author, inputText) in X_test:
            print(getFeatures(inputText))




# print(alldigit)
# input = text.text_to_word_sequence(X_test, filters=",.?!\"'`;:-()&$", lower=True, split=" ")

# freq = nltk.FreqDist(word for word in input)
# hapax = [key for key, val in freq.items() if val == 1]
# dis = [key for key, val in freq.items() if val == 2]
# print(freq)    

# bigr = ['th','he','in','er','an','re','nd','at','on','nt','ha','es','st' ,'en','ed','to','it','ou','ea','hi','is','or','ti','as','te','et' ,'ng','of','al','de','se','le','sa','si','ar','ve','ra','ld','ur']

# bg_count = {}
# for b in bigr:
#     bg_count[b] = 0
# input = text.text_to_word_sequence(data, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
# for word in input:
#     for i in range(0, len(word)-1):
#         bg = (word[i:i+2]).lower()
#         if bg in bigr:
#             bg_count[bg] = bg_count[bg] + 1
# total = sum(list(bg_count.values()))
