# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:47:25 2019

@author: Skanda
"""

import numpy as np
import pickle


def split(string):
    tokenized = []
    flag = 1
    for word in string.split(' '):
        if '"' in word:
            if flag == 1:
                tokenized.append(word)
                flag = 0
                continue
            if flag == 0:
                tokenized[-1] += ' '
                tokenized[-1] += word
                flag = 1
                continue
        if flag == 1:
            tokenized.append(word)
        else:
            tokenized[-1] += ' '
            tokenized[-1] += word
    return tokenized


with open('sbt_data_camel.pickle', 'rb') as fp:
    programs = pickle.load(fp)
    
tokenized_programs = [[] for i in range(len(programs))]
i = 0
for classname, program in programs.items():
    if program is None:
        continue
    tokenized_programs[i] = split(program)
    i = i + 1
    print(i)
    
del programs

word_index = {}
i = 1
for program in tokenized_programs:
   for token in program:
       if token in word_index.keys():
           continue
       else:
           word_index[token] = i
           i += 1
    
with open('word_index_camel.pickle', 'wb') as fp:
    pickle.dump(word_index, fp)
    
embedding_dim = 50
with open('word_vectors_camel_{}.pickle'.format(embedding_dim), 'rb') as fp:
        word_vectors = pickle.load(fp)
        
embeddings = np.random.randn(len(word_index) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored
for word, i in word_index.items():
    if word in word_vectors.vocab:
        embeddings[i] = word_vectors[word]
print('Null word embeddings: %d' % np.sum(np.sum(embeddings, axis=1) == 0))    
del word_vectors

with open('embedding_matrix_camel_{}.pickle'.format(embedding_dim), 'wb') as fp:
    pickle.dump(embeddings, fp)
    
max_seq_length = max(len(program) for program in tokenized_programs)   #48000   

           



