import numpy as np
import csv
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# K kao broj feature-a
k = 5   
s = str(k)

#Pravljenje mreze, 2 skrivena sloja, jedan output kao niz od 10 vrednosti gde je svaka vrednost verovatnoca za zanr
model = Sequential()

model.add(Input(shape = (k,)))
model.add(Dense(25 ,activation = 'sigmoid', use_bias = True, kernel_initializer=initializers.GlorotNormal))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

X_train = Y_train = x_test = y_test = np.array([])

#Dictionary koji ce pretvoriti string za dalju obradu u poslednji layer -> m['blues'] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
m = {'blues' : [1] + [0]*9, 'classical' : [0] + [1] + [0]*8, 'country' : [0]*2 + [1] + [0]*7, 'disco' : [0]*3 + [1] + [0]*6, 'hiphop' : [0]*4 + [1] + [0]*5, 
     'jazz' : [0]*5 + [1] + [0]*4, 'metal' : [0]*6 + [1] + [0]*3, 'pop' : [0]*7 + [1] + [0]*2, 'reggae' : [0]*8 + [1] + [0]*1, 'rock': [0]*9 + [1]}

t = {'blues' : 0, 'classical' : 1, 'country' : 2, 'disco' : 3, 'hiphop' : 4, 
     'jazz' : 5, 'metal' : 6, 'pop' : 7, 'reggae' : 8, 'rock': 9}

class_names=['Blues','Classical','Country','Disco','Hiphop','Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Ocitavanje fajla
with open('Fingerprint_Method/fingerprint_features.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line_cnt = -1
    test_cnt = 0
    train_cnt = 0
    for row in csv_reader:
        line_cnt += 1
        if (line_cnt == 0):
            continue

        #S obzirom da svaki zanr ima 100 semplova, prvih 90 uzimam za train, poslednjih 10 za test
        # row[i+2] predstavlja frekvencijske peak-ove, a row[0] predstavlja string
        if ((line_cnt % 100 == 0) or (line_cnt % 100 > 90)):
            test_cnt += 1
            for i in range(0,k):
                x_test = np.append(x_test,float(row[i+2])*35)
            y_test = np.append(y_test,m[row[0]])
                
        else:
            train_cnt += 1
            for i in range(0,k):
                X_train = np.append(X_train,float(row[i+2])*35)
            Y_train = np.append(Y_train,m[row[0]])

# Pretvaram ih u matrice zarad inputa u mrezu
x_test = np.reshape(x_test, (test_cnt, k))
trening = np.reshape(X_train, (train_cnt, k))

y_test = np.reshape(y_test, (test_cnt, 10))
testing = np.reshape(Y_train, (train_cnt, 10))

X_train, X_val, Y_train, Y_val = train_test_split(trening, testing, test_size=0.176470588, random_state=42)


# Treniranje
es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200, mode='min', baseline=None, restore_best_weights=True)
model.fit(X_train, Y_train, epochs = 100, callbacks = [es],  validation_data = (X_val, Y_val), shuffle = True)

train_ans = model.predict(X_train)
valid_ans = model.predict(X_val)
test_ans = model.predict(x_test)

# Provera da li se resenje poklapa 
correct_answer = 0
y_real = []
y_pred = []

for i in range(0,100):
    x = np.argmax(test_ans[i])
    y = np.argmax(y_test[i])
    if x==y:
        correct_answer += 1
    
    y_real.append(x)
    y_pred.append(y)

print(correct_answer,"%")

test_cm = confusion_matrix(y_real,y_pred)
train_cm = confusion_matrix(np.argmax(Y_train, axis = 1), np.argmax(train_ans, axis = 1))
val_cm = confusion_matrix(np.argmax(Y_val, axis = 1), np.argmax(valid_ans, axis = 1))

train_acc = 0
train_row, train_cols = Y_train.shape
for i in range(0,(train_row)):
    train_acc += (np.argmax(Y_train[i]) == np.argmax(train_ans[i]))
train_acc = train_acc / train_row * 100

val_row, val_cols = Y_val.shape
val_acc = 0
for i in range(0,val_row):
    val_acc += (np.argmax(Y_val[i]) == np.argmax(valid_ans[i]))
val_acc = val_acc / val_row * 100


plt.figure(0)
plot_confusion_matrix(conf_mat = test_cm, figsize = (10,10), class_names=class_names, colorbar=True)
plt.title('Fingerprint Method: Confusion Matrix of Test Data \n Precision: ' + str(correct_answer) + '% \n')

plt.figure(1)
plot_confusion_matrix(conf_mat = train_cm, figsize = (10,10), class_names=class_names, colorbar=True)
plt.title('Fingerprint Method: Confusion Matrix of Train Data \n Precision: ' + str(train_acc) + '% \n')

plt.figure(2)
plot_confusion_matrix(conf_mat = val_cm, figsize = (10,10), class_names=class_names, colorbar=True)
plt.title('Fingerprint Method: Confusion Matrix of Validation Data \n Precision: ' + str(val_acc) + '% \n')

plt.show()