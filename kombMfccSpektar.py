import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import scipy.fftpack as fourier 
import random
import pandas as pd
import os, csv
from numpy.core.fromnumeric import shape
from numpy.core.numeric import correlate
import pydub
import numpy as np
import scipy
from audiolazy import *
from scipy.fftpack import dct
from python_speech_features import mfcc

data = pd.read_csv('data.csv')
paths = np.array(data['path'])
genres = np.array(data['genre'])

brojmfcc = 12
brojSpektar = 64
cep_lifter = 22
n = brojmfcc + brojSpektar

features = np.zeros((1,n))

for song_path in paths:
    # Reading audio
    Fs, signal = scipy.io.wavfile.read(song_path)

    mfccKoef = mfcc(signal, samplerate = Fs,numcep = brojmfcc,highfreq=Fs/2,winfunc = np.hamming)
    mfccKoef = np.sum(mfccKoef, axis = 0)
    mfccKoef = np.reshape(mfccKoef, (1,len(mfccKoef)) )
    

    furije = abs(fourier.fft(signal))

    furije = furije[:len(furije)//2]


    furije = furije[0:len(furije)//32 *32] #odsecemo poslednje elemente niza tako da duzina bude deljiva sa 32
   
    logaritam = np.log(furije)
    duzinaSegmenta = len(furije)//32

    furije = np.reshape(furije, (32, duzinaSegmenta) )
    logaritam = np.reshape(logaritam, (32, duzinaSegmenta) )

    usrednjeno = np.array([ sum(x) // duzinaSegmenta for x in furije ])
    usrednjeno = np.reshape(usrednjeno, (1, len(usrednjeno)) )

    usrednjenLogaritam = np.array([ sum(x) / duzinaSegmenta for x in logaritam ])
    usrednjenLogaritam = np.reshape(usrednjenLogaritam, (1, len(usrednjenLogaritam)) )

    red = np.concatenate((mfccKoef, usrednjeno, usrednjenLogaritam), axis = 1)
    
    features = np.concatenate((features, red), axis = 0)

features = features[1:]
print(features.shape)

header = []
header.append('genre')
for i in range(n):
    header.append('feature'+str(i))


with open("mfccIspektar/mfccIspektarGtzan.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) 
    for i in range(len(genres)):
        x = genres[i]
        genre = np.array([x])
        feature = features[i]
        row = np.concatenate((genre, feature))
        writer.writerow(row)