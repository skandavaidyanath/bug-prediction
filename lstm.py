# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 01:08:39 2019

@author: Skanda
"""

import numpy as np
import pandas as pd
import pickle
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Model variables
n_hidden = 50
batch_size = 32
n_epoch = 10

results = []

max_seq_length = 48000

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


def fit_tokenizer(text):
    with open('word_index_camel.pickle', 'rb') as fp:
        word_index = pickle.load(fp)
    list_of_tokens = []
    for word in text:
        list_of_tokens.append(word_index[word])
    while(len(list_of_tokens) < max_seq_length):
        list_of_tokens.append(0)
    return list_of_tokens


def lstm(X_train, y_train, X_train_text, embedding_size):
    with open('embedding_matrix_camel_{}.pickle'.format(embedding_size), 'rb') as fp:
        embeddings = pickle.load(fp)
    main_input = Input(shape=(max_seq_length,), dtype='int32', name='main_input')
    x = Embedding(len(embeddings), embedding_size, weights=[embeddings], input_length=max_seq_length, trainable=False)(main_input)
    lstm_out = LSTM(n_hidden)(x)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = Input(shape=(20,), name='aux_input')
    x = concatenate([lstm_out, auxiliary_input])
    x = Dense(10, activation='relu')(x)
    x = Dense(6, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit([X_train_text, X_train], [y_train, y_train], epochs=n_epoch, batch_size=batch_size)
    model.save('lstm_camel_10epochs_{}.h5'.format(embedding_size))
    
    
    
def main():
    '''
    data = pd.read_csv('camel-1.6.csv')
    with open('missing_camel.pickle', 'rb') as fp:
        missing = pickle.load(fp)
    data = data.drop(data.index[missing])	
    X = data.iloc[:,:23]
    y = data.iloc[:,23]
    y[y > 0] = 1
    scaler = StandardScaler()
    X.iloc[:, 3:] = scaler.fit_transform(X.iloc[:, 3:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    with open('sbt_data_camel.pickle', 'rb') as fp:
        programs = pickle.load(fp)
    X_train_text = [fit_tokenizer(split(programs[classname])) for classname in X_train['name.1']]
    X_test_text = [fit_tokenizer(split(programs[classname])) for classname in X_test['name.1']]
    X_train = X_train.iloc[:, 3:]
    X_test = X_test.iloc[:, 3:]
    y_train = y_train.values
    y_test = y_test.values
    X_train_text = np.array(X_train_text)
    X_test_text = np.array(X_test_text)
    np.save('X_train_text.npy', X_train_text)
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test_text.npy', X_test_text)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    lstm(X_train, y_train, X_train_text, 50)
    '''
    model = load_model('lstm_camel_10epochs_50.h5')
    X_test_text = np.load('X_test_text.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    preds0, preds1 = model.predict([X_test_text, X_test], verbose=1)
    preds0[preds0 >= 0.5] = 1
    preds0[preds0 < 0.5] = 0
    preds1[preds1 >= 0.5] = 1
    preds1[preds1 < 0.5] = 0
    print('main output', accuracy_score(y_test, preds0))
    print('aux output', accuracy_score(y_test, preds1))
    
if __name__ == '__main__':
    main()
    