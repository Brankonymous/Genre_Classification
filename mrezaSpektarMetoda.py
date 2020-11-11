import numpy as np
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
#import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#ove dve linije koda su samo da uklone neki warning
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

sX_train, sX_test, sy_train, sy_test = train_test_split( sfeatures, slabels, test_size=0.2, random_state=42)
sX_train = np.array(sX_train)
sy_train = np.array(sy_train)
sX_test = np.array(sX_test)
sy_test = np.array(sy_test)


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

lX_train, lX_test, ly_train, ly_test = train_test_split( lfeatures, llabels, test_size=0.2, random_state=42)
lX_train = np.array(lX_train)
ly_train = np.array(ly_train)
lX_test = np.array(lX_test)
ly_test = np.array(ly_test)

#Trece kepstar feature-i

p3 = pd.read_csv('spektarMetoda/kepstarFeature.csv')
kfeatures = np.zeros((1,1800))
klabels = p3['genre']
klabels = np.array(klabels)
for i in range(12):
    colName = 'feature'+str(i)
    column = np.array(p3[colName])
    column = np.reshape(column, (1,len(column)))
    kfeatures = np.concatenate((kfeatures,column),axis = 0)
kfeatures = kfeatures[1:]


temp = np.zeros((1,12))
for i in range(1800):
    red = np.array([])
    for j in range(12):
        red = np.append(red, kfeatures[j][i])
    red = np.reshape(red, (1,len(red)))
    temp = np.concatenate((temp, red), axis = 0)
kfeatures = np.copy(temp[1:])

#print(kfeatures)

temp = np.zeros((1800,6))
i = 0
for x in klabels:
    temp[i] = m[x]
    i+=1
klabels = np.copy(temp)
#print(klabels)
kfeatures = kfeatures * 10
kX_train, kX_test, ky_train, ky_test = train_test_split( kfeatures, klabels, test_size=0.2, random_state=42)
kX_train = np.array(kX_train)
ky_train = np.array(ky_train)
kX_test = np.array(kX_test)
ky_test = np.array(ky_test)



def getModelF1(): 
    model = Sequential()
    model.add(Input(shape = (32,)))
    model.add(Dense(25 ,activation = 'sigmoid', use_bias = True, kernel_initializer=initializers.GlorotNormal))
    model.add(Dense(6, activation = 'softmax'))
    opt = tf.optimizers.Adam()

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def getModelF2():
    model = Sequential()
    model.add(Input(shape = (32,)))
    model.add(Dense(25 ,activation = 'sigmoid', use_bias = True, kernel_initializer=initializers.GlorotNormal))
    model.add(Dense(6, activation = 'softmax'))
    opt = tf.optimizers.Adam()

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def getModelF3():
    model = Sequential()
    model.add(Input(shape = (12,)))
    model.add(Dense(25 ,activation = 'sigmoid', use_bias = True, kernel_initializer=initializers.GlorotNormal))
    #model.add(Dense(16 ,activation = 'relu', use_bias = True, kernel_initializer=initializers.GlorotNormal))
    model.add(Dense(6, activation = 'softmax'))
    opt = tf.optimizers.Adam()

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = getModelF2()
model.summary()
es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200, mode='min', baseline=None, restore_best_weights=True)
#es = EarlyStopping(monitor = 'val_accuracy', mode = 'max',patience = 50, restore_best_weights = True)
#history = model.fit(lX_train, ly_train, epochs = 300, callbacks = [es],  validation_split = 0.176470588, shuffle = True)   
#print(history.history.keys())
#model.save_weights('spektarMetoda/modelF2.h5')
model.load_weights('spektarMetoda/modelF2.h5')

rez = model.predict(lX_test)

y_actu = np.array([])
y_pred = np.array([])

correct_answer = 0
for i in range(0, len(ly_test)):
    x = np.argmax(rez[i])
    y_pred = np.append(y_pred, x) 
    y = np.argmax(ly_test[i])
    y_actu = np.append(y_actu, y)
    if x==y:
        correct_answer += 1
correct_answer/=len(ly_test)
correct_answer *= 100
print(correct_answer,"%")


print(accuracy_score(y_actu, y_pred))

cm = confusion_matrix(y_actu,y_pred)
plot_confusion_matrix(conf_mat=cm, figsize = (10,10),class_names=['Klasika','Folk','House','Jazz','RnB','Rock'])
plt.show()

#plt.plot(history.history['val_loss'])
#plt.show()




