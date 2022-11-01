from distutils.errors import LinkError
import json
import os
import numpy as np
import pandas as pd
# from tqdm import tqdm

data,ctr = {'stars': [], 'text': []},1
f1 = open('./dataset/yelp/yelp.txt','w',encoding='utf-8')
with open('./dataset/yelp_academic_dataset_review.json',encoding='utf-8') as f:
    tr = f.readline()
    while tr:
        review = json.loads(tr)
        # data['stars'].append(review['stars'])
        # data['text'].append(review['text'])
        # while ctr<10:
        # print(str(ctr)+' '+str(review['stars'])+' '+review['text'])
        f1.write(str(ctr)+' '+str(review['stars'])+' '+review['text'].replace('\n',' ')+'\n')
        ctr+=1
        tr = f.readline()
    
print('Done!')
# df = pd.DataFrame(data)

# print(df.shape)
# df.head()
# df['stars'] = df['stars'].astype('category')
# df['text'] = df['text'].astype(str)
# df.to_csv('yelp_reviews.csv', index=False)
# import tensorflow as tf
# table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file="dataset\yelp\id2w.txt")
# a = table.lookup(tf.constant([1, 100], tf.int64))
# values = table.lookup(tf.constant([1, 5], tf.int64))
# # tf.tables_initializer().run()

# print(values)