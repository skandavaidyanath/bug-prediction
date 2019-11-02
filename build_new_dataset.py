# -*- coding: utf-8 -*-
"""
Created on Mon May 27 07:08:55 2019

@author: Skanda
"""

import pandas as pd
import numpy as np
import pickle
from word2vec import split


EMBEDDING_DIMS = 300

def main():
    with open('sbt_data_camel.pickle', 'rb') as fp:
        programs = pickle.load(fp)
    with open('word_vectors_camel_{}.pickle'.format(EMBEDDING_DIMS), 'rb') as fp:
        word_vectors = pickle.load(fp)
    dict_of_vectors = {}
    i = 0
    for classname, program in programs.items():
        if program is None:
            i =  i + 1
            continue
        tokenized = split(program)
        word_vector_sum = np.zeros(EMBEDDING_DIMS)
        for word in tokenized:
            word_vector_sum += word_vectors[word]
        word_vector_avg = word_vector_sum/len(tokenized)
        dict_of_vectors[classname] = word_vector_avg
        i =  i + 1
        print(i)
    df = pd.DataFrame.from_dict(dict_of_vectors, orient='index')
    df['name.1'] = df.index
    data = pd.read_csv('./camel-1.6.csv')
    with open('missing_camel.pickle', 'rb') as fp:
        missing = pickle.load(fp)
    data = data.drop(data.index[missing])
    new_data = pd.merge(data, df, on='name.1')
    new_data = new_data.iloc[:, 3:]
    new_data.to_csv('new_data_camel_{}.csv'.format(EMBEDDING_DIMS), encoding='utf-8', index=False)

    
if __name__ == '__main__':
    main()
    