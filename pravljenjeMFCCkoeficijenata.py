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


def hzToMel(a):
    return 2595 * np.log10(1 + (a / 2) / 700)
def melToHz(a):
    for i in range(len(a)):
        a[i] =  700 * (10**(a[i] / 2595) - 1)
    return a

data = pd.read_csv('mfccMetoda/nataSort.csv')
paths = np.array(data['path'])
genres = np.array(data['genre'])

brojKoef = 12
cep_lifter = 22

features = np.zeros((1,brojKoef))
i = 0

for song_path in paths:
    # Reading audio
    Fs, signal = scipy.io.wavfile.read(song_path)
    koef = mfcc(signal, samplerate = Fs,numcep = brojKoef,highfreq=Fs/2,winfunc = np.hamming)
   
    '''
    emp_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Framing
    frame_length, frame_step = int(round(0.025 * Fs)), int(round(0.01 * Fs))
    signal_length = len(emp_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length) / frame_step)))

    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(emp_signal, np.zeros((pad_signal_length - signal_length)))

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Windowing
    frames *= np.hamming(frame_length)
    nFourier = 512


    #creating power spectrum

    spektar = np.absolute(np.fft.rfft(frames, nFourier))  # Magnitude of the FFT
    powerSpectrum = ((1.0 / nFourier) * ((spektar) ** 2))

    #triangular filter

    brojFiltera = 26

    donjaGranica = 0
    gornjaGranica = hzToMel(Fs/2)

    mel = np.linspace(donjaGranica, gornjaGranica,brojFiltera+2 )
    herc = melToHz(mel) #Sve je u herc skali, al tako da je rasporedjeno ravnomerno po Mel skali

    bin = np.floor((nFourier+1)*herc / Fs) #prelomne tacke na x osi

    banka = np.zeros((brojFiltera, int(np.floor(nFourier/2+1)) ), dtype = float)

    for k in range(1,brojFiltera+1):
        levo = bin[k-1]
        sredina = bin[k]
        desno = bin[k+1]
        trenutnaBanka = banka[k-1]
        for i in range(int(levo),int(sredina)):
            trenutnaBanka[i] = (i - levo) / (sredina - levo) #normalizovana vrednost
        for i in range(int(sredina), int(desno)):
            trenutnaBanka[i] = (desno - i) / (desno - sredina)
        banka[k-1] = trenutnaBanka
    

    gotovaBanka = np.dot(powerSpectrum, banka.T)

    for i in range(len(gotovaBanka)):
        for j in range(len(gotovaBanka[i])):
            if gotovaBanka[i][j] == 0:
                gotovaBanka[i][j] = 10**(-10) #da ne bude nula
    gotovaBanka = 20 * np.log10(gotovaBanka) #u decibele

    

    koef = dct(gotovaBanka, type=2, axis=1, norm='ortho')[:, 1 : (brojKoef + 1)]
    '''

    koef = np.sum(koef, axis = 0)
    koef = np.reshape(koef, (1,len(koef)) )
    features = np.concatenate((features, koef), axis = 0)
features = features[1:]
  

    #koef -= (np.mean(koef, axis=0) + 1e-8) #mean normalization
  
    
print(features.shape)
header = []
header.append('genre')
for i in range(brojKoef):
    header.append('feature'+str(i))


with open("fingerprintImfcc/mfccSort.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) 
    for i in range(len(genres)):
        x = genres[i]
        genre = np.array([x])
        feature = features[i]
        row = np.concatenate((genre, feature))
        writer.writerow(row)