from distutils.errors import LinkError
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
df = pd.read_csv('./dataset/train.csv')
works = defaultdict(str)
for ind in df.index:
    works[df['author'][ind]] += df['review'][ind].replace('\n',' ')
# print(works['A11OTLEDSW8ZXD'])

tokenizer = RegexpTokenizer(r'\w+')
data,ctr = {'text': []},1
f1 = open('./dataset/yelp/yelp.txt','w',encoding='utf-8')
f2 = open('./dataset/yelp/id2w.txt','w',encoding='utf-8')
lt,st = defaultdict(int),set()
for a in works:
    for a1 in tokenizer.tokenize(works[a]):
        lt[a1]+=1
    f1.write(str(ctr)+' '+works[a]+'\n')
    ctr+=1
    if ctr==10000:
        break
# with open('./dataset/yelp_academic_dataset_review.json',encoding='utf-8') as f:
#     tr = f.readline()
#     while tr:
#         if ctr==10000:
#             break
#         review = json.loads(tr)
#         review['text'] = review['text'].replace('\n',' ').replace(chr(13),' ')
#         f1.write(str(ctr)+' '+str(review['stars'])+' '+review['text']+'\n')
        # for a in tokenizer.tokenize(review['text']):
        #     lt[a]+=1
#         ctr+=1
#         tr = f.readline()
ptr = sorted(lt.keys(),key = lambda x:-lt[x])
# print(ptr)
inder,my_dct = 0,{}
for a in ptr:
    f2.write(a+'\n')
    my_dct[a] = inder
    inder+=1
with open("my_dict.json", "w") as outfile:
    json.dump(my_dct, outfile)
print('Done!')