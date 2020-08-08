#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:10:41 2020

@author: uribarrix
"""

import pandas as pd
from os import listdir


# Importamos o .csv có mapa con píxeles clasificados e pómos nome ás columnas.
data_set = pd.read_csv('imaxe satelite/clasificacion.csv', sep = " ", names = ['X', 'Y', 'class'])

# Seleccionamos só os píxeles clasificados.
data_set = data_set.loc[data_set['class'] >= 0]

# Damos valor 1 aos que teñan un valor maior que 1 na columna Z.
data_set[data_set['class'] > 1] = 1

# Vamos construir o dataset coas bandas nas columnas para os píxeles clasificados.
folder = 'imaxe satelite/recortes/'


band_list = listdir(folder)
band_list.sort()

for band in band_list:
    banda = pd.read_csv(folder+band, sep = " ", names = ['X', 'Y', band[0:3]])
    data_set = pd.merge(data_set, banda, on=['X', 'Y'], how = "inner")

    
# Dividimos o data_set
X = data_set.iloc[:, 3:].values
y = data_set.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalamos as variables entorno a 0 para que non exista distorsión por estaren a escalas distintas.
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# ELABORAR RED NEURONAL

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier():
    # Inicializar a RNA
    classifier = Sequential()
    
    # Engadir as capas de entrada e a primeria capa oculta
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                         activation = 'relu', input_dim = 12))
    classifier.add(Dropout(p=0.2))
    
    # Engadir a segunda capa oculta
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                         activation = 'relu'))
    classifier.add(Dropout(p=0.2))
    
    #Capa de saída
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                         activation = 'sigmoid'))
    
    # Compilar a RNA
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy",
                       metrics = ["accuracy"])
    # Devolder o clasificador
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 1, epochs = 1000)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

