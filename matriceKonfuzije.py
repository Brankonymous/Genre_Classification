import numpy as np
from numpy.core.records import array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import random
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix

import os

from tensorflow_core import optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#mapiranje zanrova u brojeve

m = {'classical' : [1] + [0]*5, 'folk' : [0] + [1] + [0]*4, 'house' : [0]*2 + [1] + [0]*3, 
'jazz' : [0]*3 + [1] + [0]*2, 'rnb' : [0]*4 + [1] + [0]*1, 'rock' : [0]*5 + [1]}

t = {'classical' : 0, 'folk' : 1, 'house' : 2, 'jazz' : 3, 'rnb' : 4, 
     'rock' : 5}
#ucitavanje podataka 

# Prvo spektar feature-i

p1 = pd.read_csv('spektarMetoda/spektarFeature.csv')
sfeatures = np.zeros((1,1800))
slabels = p1['genre']
slabels = np.array(slabels)

for i in range(32):
    colName = 'feature'+str(i)
    column = np.array(p1[colName])       
    column = np.reshape(column, (1,len(column)) )
    sfeatures = np.concatenate((sfeatures,column),axis = 0)

sfeatures = sfeatures[1:] #izbacivanje kolone s nulama
temp = np.zeros((1,32))
for i in range(1800):
    red = np.array([])
    for j in range(32):
        red = np.append(red, sfeatures[j][i])
    red = np.reshape(red, (1,len(red)))
    temp = np.concatenate((temp, red), axis = 0)
sfeatures = np.copy(temp[1:])

temp = np.zeros((1800,6))
i = 0
for x in slabels:
    temp[i] = m[x]
    i+=1
slabels = np.copy(temp)

sfeatures = (sfeatures - np.min(sfeatures))
sfeatures = sfeatures/np.max(sfeatures)
sfeatures = sfeatures * (25+6)

print(sfeatures.shape, slabels.shape)

# Drugo logaritam spektra feature-i


p2 = pd.read_csv('spektarMetoda/logFeature.csv')
lfeatures = np.zeros((1,1800))
llabels = p2['genre']
llabels = np.array(llabels)
for i in range(32):
    colName = 'feature'+str(i)
    column = np.array(p2[colName])
    column = np.reshape(column, (1,len(column)))
    lfeatures = np.concatenate((lfeatures,column),axis = 0)
lfeatures = lfeatures[1:]

temp = np.zeros((1,32))
for i in range(1800):
    red = np.array([])
    for j in range(32):
        red = np.append(red, lfeatures[j][i])
    red = np.reshape(red, (1,len(red)))
    temp = np.concatenate((temp, red), axis = 0)
lfeatures = np.copy(temp[1:])

temp = np.zeros((1800,6))
i = 0
for x in llabels:
    temp[i] = m[x]
    i+=1
llabels = np.copy(temp)

lfeatures = (lfeatures - np.min(lfeatures))
lfeatures = lfeatures/np.max(lfeatures)
lfeatures = lfeatures * (25+6)

print(lfeatures.shape, llabels.shape)

slfeatures = np.concatenate((sfeatures, lfeatures), axis = 1)
print(slfeatures.shape)

sX_trainVal, sX_test, sy_trainVal, sy_test = train_test_split( slfeatures, llabels, test_size=0.15, random_state=42)

n = len(sX_trainVal)

sX_train = sX_trainVal[:int(n*(1 - 0.1764))]
sX_val = sX_trainVal[int(n*(1 - 0.1764)):]

sy_train = sy_trainVal[:int(n*(1 - 0.1764))]
sy_val = sy_trainVal[int(n*(1 - 0.1764)):]


sX_train = np.array(sX_train)
sy_train = np.array(sy_train)
sX_test = np.array(sX_test)
sy_test = np.array(sy_test)
sX_val = np.array(sX_val)
sy_val = np.array(sy_val)


def getModelF2(): 
    model = Sequential()
    model.add(Input(shape = (64,)))
    model.add(Dense(25 ,activation = 'sigmoid', use_bias = True, kernel_initializer=initializers.GlorotNormal))
    model.add(Dense(6, activation = 'softmax'))
    opt = tf.optimizers.Adam()

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = getModelF2()

es = EarlyStopping(monitor = 'val_accuracy', mode = 'max',patience = 50, restore_best_weights = True)
#history = model.fit(sX_train, sy_train, epochs = 300, callbacks = [es],  validation_data = ( sX_val, sy_val), shuffle = True)
#model.save_weights("spektarMetoda/modelF12.h5")
model.load_weights("spektarMetoda/modelF12.h5")

#test data matrica 
rez = model.predict(sX_test)
y_pred = np.array([])
y_actu = np.array([])

for i in range(0, len(sy_test)):
    x = np.argmax(rez[i])
    y_pred = np.append(y_pred, x) 
    y = np.argmax(sy_test[i])
    y_actu = np.append(y_actu, y)
    

print(accuracy_score(y_actu, y_pred))

cm = confusion_matrix(y_actu,y_pred)
print(cm)



plot_confusion_matrix(conf_mat=cm, figsize = (10,10),class_names=['Klasika','Folk','House','Jazz','RnB','Rock'])
plt.show()

