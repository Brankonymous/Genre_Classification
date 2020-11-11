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


m = {'blues' : [1] + [0]*9, 'classical' : [0] + [1] + [0]*8, 'country' : [0]*2 + [1] + [0]*7, 'disco' : [0]*3 + [1] + [0]*6, 'hiphop' : [0]*4 + [1] + [0]*5, 
     'jazz' : [0]*5 + [1] + [0]*4, 'metal' : [0]*6 + [1] + [0]*3, 'pop' : [0]*7 + [1] + [0]*2, 'reggae' : [0]*8 + [1] + [0]*1, 'rock': [0]*9 + [1]}

t = {'blues' : 0, 'classical' : 1, 'country' : 2, 'disco' : 3, 'hiphop' : 4, 
     'jazz' : 5, 'metal' : 6, 'pop' : 7, 'reggae' : 8, 'rock': 9}

class_names=['Blues','Classical','Country','Disco','Hiphop','Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']


p = pd.read_csv('mfccIspektarGtzan.csv')     

brojKoef = 76
brojIzlaza = 10

labels = p['genre']
labels = np.array(labels)


features = np.array([])
readData = np.zeros((1,len(labels)))

for i in range(brojKoef):
    colName = 'feature'+str(i)
    column = np.array(p[colName])       
    column = np.reshape(column, (1,len(column)) )
    readData = np.concatenate((readData,column),axis = 0)

readData = readData[1:]

features = readData.T

temp = np.zeros((len(labels),brojIzlaza))
i = 0
for x in labels:
    temp[i] = m[x]
    i+=1
labels = np.copy(temp)

features = (features - np.min(features))
features = features/np.max(features)
features = features * (25+brojIzlaza)



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
    model.add(Input(shape = (brojKoef,)))
    model.add(Dense(50 ,activation = 'sigmoid', use_bias = True, kernel_initializer=initializers.GlorotNormal))
    model.add(Dense(brojIzlaza, activation = 'softmax'))
    opt = tf.optimizers.Adam(learning_rate = 0.001)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = getModel()

es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 50, restore_best_weights = True)

#history = model.fit(x_train, y_train, epochs = 500, callbacks = [es],  validation_data = ( x_val, y_val), shuffle = True)
#model.save_weights("mfccIspektarGtzan40.h5")
model.load_weights("mfccIspektarGtzan50.h5")

rez = model.predict(x_train)
y_pred = np.array([])
y_actu = np.array([])

for i in range(0, len(y_train)):
    x = np.argmax(rez[i])
    y_pred = np.append(y_pred, x) 
    y = np.argmax(y_train[i])
    y_actu = np.append(y_actu, y)
    
print(accuracy_score(y_actu, y_pred))

cm = confusion_matrix(y_actu,y_pred)

plot_confusion_matrix(conf_mat=cm, figsize = (10,10),class_names=class_names)
plt.show()


