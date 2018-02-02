# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:29:27 2018

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#turn category into ordinal number
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#index of the column becomes parameter
#make dummy variable with onehotencoder
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

#initiliasing ANN
classifier = Sequential()

#adding input layer and the first hidden layer
classifier.add(Dense(6, kernel_initializer='uniform',  activation = 'relu', input_shape = (11,)))

#adding second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform',  activation = 'relu'))

#adding output layer
classifier.add(Dense(1, kernel_initializer='uniform',  activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

#predicting the test set results
y_pred = classifier.predict(X_test)
#will return True if above threshold and otherwise
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
