# -*- coding: utf-8 -*-
"""
Created on Mon May 27 02:18:55 2019

@author: Skanda
"""

from gensim.models import Word2Vec
import pickle


EMBEDDING_DIMS = 300

def get_word2vec(tokenized_programs, embedding_dim):
    model = Word2Vec(tokenized_programs, min_count=1, size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors


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

def main():
    with open('sbt_data_camel.pickle', 'rb') as fp:
        programs = pickle.load(fp)
    tokenized_programs = [[] for i in range(len(programs))]
    i = 0
    missing = []
    fp = open('where_camel.txt', 'w+')
    for classname, program in programs.items():
        if program is None:
            missing.append(i)
            i =  i + 1
            fp.write(str(classname))
            fp.write('\n')
            continue
        tokenized_programs[i] = split(program)
        i = i + 1
        print(i)
    fp.close()
    print(len(missing))
    with open('missing_camel.pickle', 'wb') as fp:
        pickle.dump(missing, fp)
    word_vectors = get_word2vec(tokenized_programs, EMBEDDING_DIMS)
    print(len(word_vectors.vocab))
    with open('word_vectors_camel_{}.pickle'.format(EMBEDDING_DIMS), 'wb') as fp:
        pickle.dump(word_vectors, fp)
    

    
if __name__ == '__main__':
    main()
    
        