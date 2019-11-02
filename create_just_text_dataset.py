# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:52:28 2019

@author: Skanda
"""

import pandas as pd
import numpy as np
import pickle
from word2vec import split


EMBEDDING_DIMS = 300

def main():
    with open('sbt_data.pickle', 'rb') as fp:
        programs = pickle.load(fp)
    with open('word_vectors_{}.pickle'.format(EMBEDDING_DIMS), 'rb') as fp:
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
    df['classname '] = df.index
    data = pd.read_csv('./lucene/lucene/single-version-ck-oo.csv', sep=';')
    with open('missing.pickle', 'rb') as fp:
        missing = pickle.load(fp)
    data = data.drop(data.index[missing])
    data = pd.concat([data['classname '], data[' bugs ']], axis=1)
    new_data = pd.merge(df, data, on='classname ')
    new_data = new_data.drop('classname ', axis=1)
    new_data.to_csv('just_text_data_{}.csv'.format(EMBEDDING_DIMS), encoding='utf-8', index=False)

    
if __name__ == '__main__':
    main()