# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:49:31 2019

@author: Skanda
"""

import pandas as pd
import numpy as np
import pickle
from word2vec_combined import split


EMBEDDING_DIMS = 50

def main():
    df1 = pd.read_csv('./camel-1.6.csv')
    #df2 = pd.read_csv('./velocity-1.6.csv')
    df3 = pd.read_csv('./lucene-2.4.csv')
    df4 = pd.read_csv('./ant-1.7.csv')
    df_combined = pd.concat([df1, df3, df4], axis=0)
    df_combined.to_csv('combined_minus_velocity.csv', encoding='utf-8')
    with open('sbt_data_combined_minus_velocity.pickle', 'rb') as fp:
        programs = pickle.load(fp)
    with open('word_vectors_combined_minus_velocity_{}.pickle'.format(EMBEDDING_DIMS), 'rb') as fp:
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
    data = pd.read_csv('./combined_minus_velocity.csv')
    with open('missing_combined_minus_velocity.pickle', 'rb') as fp:
        missing = pickle.load(fp)
    data = data.drop(data.index[missing])
    new_data = pd.merge(data.iloc[:, 3:], df, on='name.1')
    new_data = new_data.drop(['name.1'], axis=1)
    new_data.to_csv('data_embeddings_combined_minus_velocity_{}.csv'.format(EMBEDDING_DIMS), encoding='utf-8', index=False)
    new_data = pd.merge(df, pd.concat([data['name.1'], data['bug']], axis=1), on='name.1')
    new_data = new_data.drop(['name.1'], axis=1)
    new_data.to_csv('just_text_data_combined_minus_velocity_{}.csv'.format(EMBEDDING_DIMS), encoding='utf-8', index=False)

    
if __name__ == '__main__':
    main()