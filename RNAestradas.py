#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:47:14 2020

@author: uribarrix
"""

import pandas as pd
from os import listdir

# Importamos ficheiro .csv do mapa cos píxeles clasificados e pómos nome as columnas. 
data_set = pd.read_csv('imaxe satelite/clasificacion.csv', sep = " ", names = ['X', 'Y', 'class'])

# Seleccionamos só os píxeles clasificados.
data_set = data_set.loc[data_set['class'] >= 0]

# Damos valor 1 aos que teñan un valor maior que 1 na columna 'class' (ás veces tecleamos '11').
data_set[data_set['class'] > 1] = 1

# Vamos construir o dataset coas bandas nas columnas para os píxeles clasificados.
folder = 'imaxe satelite/recortes/'
folder2 = 'imaxe satelite/xixon_csv/'

# Creamos listas cos nomes dos ficheiros das imaxes das distintas bandas
band_list = listdir(folder)
band_list.sort()
band_list2 = listdir(folder2)
band_list2.sort()

# Creamos o dataframe que contén as coordendas de toda a imaxe
imaxe = pd.read_csv(folder2+'B01.csv', sep = " ", names = ['X', 'Y', 'class']).iloc[:, 0:2]

# Creamos o data set que contén as 12 bandas da imaxe no recorte a clasificar
for band in band_list2:
  banda = pd.read_csv(folder2+band, sep = " ", names = ['X', 'Y', band[0:3]], engine = "python")
  imaxe = pd.merge(imaxe, banda, on=['X', 'Y'], how = "inner")

# Engadimos as bandas ao data set dos puntos clasificados a man.
for band in band_list:
    banda = pd.read_csv(folder+band, sep = " ", names = ['X', 'Y', band[0:3]])
    data_set = pd.merge(data_set, banda, on=['X', 'Y'], how = "inner")

    
# Dividimos o data_set en Matriz de características e vector de clasificación
X = data_set.iloc[:, 3:].values
y = data_set.iloc[:, 2].values

# O data set do recorte a clasificar non ten vector de clasificación
X2 = imaxe.iloc[:, 2:].values

# Dividimos o data_set en conxuntos de adestramento e test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalamos as variables entorno a 0 para que non exista distorsión por estaren a escalas distintas.
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X2_scaled = sc_X.transform(X2)


# ELABORAR RED NEURONAL

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Inicializar a RNA
classifier = Sequential()

# Engadir as capas de entrada e a primeria capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 12))
classifier.add(Dropout(p=0.1))

# Engadir a segunda capa oculta
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', 
                     activation = 'relu'))
classifier.add(Dropout(p=0.1))

# Engadir a terceira capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', 
                     activation = 'relu'))

classifier.add(Dropout(p=0.1))
#Capa de saída
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

# Compilar a RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy",
                   metrics = ["accuracy"])

# Devolver o classificador
classifier.fit(X_train, y_train, epochs = 200, batch_size = 8)

# Elaborar predicción do conxunto de test.
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Elaborar a matriz de confusión.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Facer predicción sobre o data set da imaxe a clasificar (clasificación)
import numpy as np
imaxe_clasificada = classifier.predict(X2_scaled)

# Os valores son decimais entre 0 e 1 e queremos que sexan 0 ou 1. Redondeamos a 0 decimais.
imaxe_clasificada = np.round(imaxe_clasificada, 0)

# Transformamos o numpy array nun dataframe con unha soa columna.
imaxe_clasificada = pd.DataFrame(imaxe_clasificada, columns = ['class'])

# Concatenamos o dataframe da clasificación o final do data set da imaxe a clasificar.
imaxe_completa = pd.concat([imaxe, imaxe_clasificada], axis = 1)

# Seleccionamos as columnas que van formar a capa clasificada (coordenadas e clasificación)
resultado_final = imaxe_completa.iloc[:, [0, 1, 14]]

# Para cargar o .csv no QGIS non pode ter cabeceira, nen columna de índices.
resultado_final.to_csv('capa_clasificada.csv', header = False, index = False)
