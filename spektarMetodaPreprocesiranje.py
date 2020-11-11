import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import scipy.fftpack as fourier 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import random
import pandas as pd
import csv

data = pd.read_csv('Combining_Features/nataSort.csv')



paths = np.array(data['path'])
genres = np.array(data['genre'])
i = 0

features = np.zeros((1,32))
logFeatures = np.zeros((1,32))
kepFeatures =  np.zeros((1,12))
for path in paths: 
    i+=1
    Fs, pesma = wavfile.read(path)
    

    furije = abs(fourier.fft(pesma))

    furije = furije[:len(furije)//2]


    furije = furije[0:len(furije)//32 *32] #odsecemo poslednje elemente niza tako da duzina bude deljiva sa 32
   
    logaritam = np.log(furije)
    duzinaSegmenta = len(furije)//32
    #kepstar = fourier.ifft(np.log(fourier.fft(pesma)))
    #kepstar = abs(kepstar)
    #kepstar = kepstar[:len(kepstar)//2]
    #kepstar = kepstar[:len(kepstar)//12 * 12]
    #duzinaKepstra = len(kepstar)//12
    #if i==1:
     #   print(len(kepstar),len(furije))
    
    #naredni redovi su segmentacija na 32 dela
    furije = np.reshape(furije, (32, duzinaSegmenta) )
    logaritam = np.reshape(logaritam, (32, duzinaSegmenta) )
    #kepstar = np.reshape(kepstar, (12, duzinaKepstra) )

    usrednjeno = np.array([ sum(x) // duzinaSegmenta for x in furije ])
    usrednjeno = np.reshape(usrednjeno, (1, len(usrednjeno)) )

    usrednjenLogaritam = np.array([ sum(x) / duzinaSegmenta for x in logaritam ])
    usrednjenLogaritam = np.reshape(usrednjenLogaritam, (1, len(usrednjenLogaritam)) )

    #usrednjeniKepstar = np.array([sum(x) / duzinaKepstra for x in kepstar]) #uzet median jer prosek daje los rezultat usrednjavanje
    #usrednjeniKepstar = np.reshape(usrednjeniKepstar, (1, len(usrednjeniKepstar)) )

    features = np.concatenate((features,usrednjeno), axis = 0)
    logFeatures = np.concatenate((logFeatures,usrednjenLogaritam), axis = 0)
    #kepFeatures = np.concatenate((kepFeatures,usrednjeniKepstar), axis = 0)

features = features[1:]
logFeatures = logFeatures[1:]
#kepFeatures = kepFeatures[1:]

header = []
header.append('genre')
for i in range(32):
    header.append('feature'+str(i))
final = np.array([])

with open("spektarMetoda/spektarFeatureSort.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) 
    for i in range(len(genres)):
        x = genres[i]
        genre = np.array([x])
        feature = features[i]
        row = np.concatenate((genre, feature))
        writer.writerow(row)

header = []
header.append('genre')
for i in range(32):
    header.append('feature'+str(i))
final = np.array([])

with open("spektarMetoda/logFeatureSort.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) 
    for i in range(len(genres)):
        x = genres[i]
        genre = np.array([x])
        feature = logFeatures[i]
        row = np.concatenate((genre, feature))
        writer.writerow(row)

        '''
header = []
header.append('genre')
for i in range(12):
    header.append('feature'+str(i))
final = np.array([])

with open("spektarMetoda/kepstarFeatureGTZAN.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) 
    for i in range(len(genres)):
        x = genres[i]
        genre = np.array([x])
        feature = kepFeatures[i]
        row = np.concatenate((genre, feature))
        writer.writerow(row)
'''
