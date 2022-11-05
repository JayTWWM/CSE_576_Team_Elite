# class Vocabulary:
    
#     PAD_token = 0   # Used for padding short sentences
#     SOS_token = 1   # Start-of-sentence token
#     EOS_token = 2   # End-of-sentence token

#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
#         self.num_words = 3
#         self.num_sentences = 0
#         self.longest_sentence = 0

#     def add_word(self, word):
#         if word not in self.word2index:
#             # First entry of word into vocabulary
#             self.word2index[word] = self.num_words
#             self.word2count[word] = 1
#             self.index2word[self.num_words] = word
#             self.num_words += 1
#         else:
#             # Word exists; increase word count
#             self.word2count[word] += 1
            
#     def add_sentence(self, sentence):
#         sentence_len = 0
#         for word in sentence.split(' '):
#             sentence_len += 1
#             self.add_word(word)
#         if sentence_len > self.longest_sentence:
#             # This is the longest sentence
#             self.longest_sentence = sentence_len
#         # Count the number of sentences
#         self.num_sentences += 1

#     def to_word(self, index):
#         return self.index2word[index]

#     def to_index(self, word):
#         return self.word2index[word]

# voc = Vocabulary("test")

# corpus = ['This is the first sentence....',
#           'This is the second!!!',
#           'There is no sentence in this corpus longer than this (one).',
#           'My dog is named Patrick.']
# print(corpus)

# for sent in corpus:
#   voc.add_sentence(sent)

# print('Token 4 corresponds to token:', voc.to_word(4))
# print('Token "this" corresponds to index:', voc.to_index('this'))

# for word in range(voc.num_words):
#     print(voc.to_word(word))
# import nltk
# nltk.download('punkt')
# f=open('./dataset/yelp/yelp1.txt','rU',encoding='utf-8')
# raw=f.read()
# tokens = nltk.word_tokenize(raw)
# text = nltk.Text(tokens)
# print(tokens)
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
print(tokenizer.tokenize('Eighty-seven miles to go, go yet.  Onward!'))