# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 00:12:24 2019

@author: Skanda
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split


def run_normal():   
    data = pd.read_csv('./camel-1.6.csv')	
    data = data.iloc[:, 3:]
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y[y > 0] = 1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier = Sequential()
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 20))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    classifier.save('normal_camel.h5')
    print(accuracy_score(y_test, y_pred))
    
    
def run_embeddings(embedding_size):
    new_data = pd.read_csv('new_data_camel_{}.csv'.format(embedding_size))
    y = new_data['bug']
    y[y > 0] = 1
    X = new_data.drop(['bug'], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier = Sequential()
    classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = embedding_size+20))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    classifier.save('new_data_model_camel_{}.h5'.format(embedding_size))
    print(accuracy_score(y_test, y_pred))
    
    
def main():
    embedding_size = 300
    #run_normal()
    run_embeddings(embedding_size)
    #print(results)
    '''
    new_data = pd.read_csv('just_text_data_{}.csv'.format(embedding_size))
    y = new_data.iloc[:, embedding_size]
    y[y > 0] = 1
    X = new_data.drop(new_data.columns[[embedding_size]], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = load_model('just_text_model_{}.h5'.format(embedding_size))
    y_pred = model.predict(X)
    y_pred = (y_pred >= 0.5)
    print(accuracy_score(y, y_pred))
    '''
    
    
if __name__ == '__main__':
    main()
