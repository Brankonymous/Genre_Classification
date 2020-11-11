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


m = {'classical' : [1] + [0]*5, 'folk' : [0] + [1] + [0]*4, 'house' : [0]*2 + [1] + [0]*3, 
'jazz' : [0]*3 + [1] + [0]*2, 'rnb' : [0]*4 + [1] + [0]*1, 'rock' : [0]*5 + [1]}

t = {'classical' : 0, 'folk' : 1, 'house' : 2, 'jazz' : 3, 'rnb' : 4, 
     'rock' : 5}

p = pd.read_csv('mfccMetoda/mfccGotovi.csv')

brojKoef = 12

labels = p['genre']
labels = np.array(labels)


features = np.array([])
readData = np.zeros((1,1800))

for i in range(brojKoef):
    colName = 'feature'+str(i)
    column = np.array(p[colName])       
    column = np.reshape(column, (1,len(column)) )
    readData = np.concatenate((readData,column),axis = 0)

readData = readData[1:]

features = readData.T

temp = np.zeros((1800,6))
i = 0
for x in labels:
    temp[i] = m[x]
    i+=1
labels = np.copy(temp)

#normalizacija 

features = (features - np.min(features))
features = features/np.max(features)
features = features * (25+6)


#podela na train, val i test

x_trainVal, x_test, y_trainVal, y_test = train_test_split( features, labels, test_size=0.15, random_state=42)

n = len(x_trainVal)

x_train = x_trainVal[:int(n*(1 - 0.1764))]
x_val = x_trainVal[int(n*(1 - 0.1764)):]

y_train = y_trainVal[:int(n*(1 - 0.1764))]
y_val = y_trainVal[int(n*(1 - 0.1764)):]


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)


def getModel(): 
    model = Sequential()
    model.add(Input(shape = (12,)))
    model.add(Dense(25 ,activation = 'sigmoid', use_bias = True, kernel_initializer=initializers.GlorotNormal))
    model.add(Dense(6, activation = 'softmax'))
    opt = tf.optimizers.Adam(learning_rate = 0.001)
    #opt = tf.optimizers.Adadelta(learning_rate = 0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = getModel()

es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 50, restore_best_weights = True)

#history = model.fit(x_train, y_train, epochs = 500, callbacks = [es],  validation_data = ( x_val, y_val), shuffle = True)
#model.save_weights("mfccMetoda/mfccNata-25.h5")
model.load_weights("mfccMetoda/mfccNata-25.h5")

rez = model.predict(x_test)
y_pred = np.array([])
y_actu = np.array([])

for i in range(0, len(y_test)):
    x = np.argmax(rez[i])
    y_pred = np.append(y_pred, x) 
    y = np.argmax(y_test[i])
    y_actu = np.append(y_actu, y)
    

print(accuracy_score(y_actu, y_pred))

#plt.plot(history.history['val_loss'])
#plt.show()

cm = confusion_matrix(y_actu,y_pred)
print(cm)


plot_confusion_matrix(conf_mat=cm, figsize = (10,10),class_names=['Klasika','Folk','House','Jazz','RnB','Rock'])
plt.show()
