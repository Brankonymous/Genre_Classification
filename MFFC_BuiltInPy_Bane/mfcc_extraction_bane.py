import csv
from python_speech_features import mfcc
import scipy
import numpy as np
import pandas as pd
from scipy.io import wavfile

data = pd.read_csv('data.csv')
paths = np.array(data['path'])
genres = np.array(data['genre'])

features = []
gen = []
cnt = 0

for song_path in paths:
    Fs, signal = scipy.io.wavfile.read(song_path)
    coeffs = mfcc(signal, samplerate = Fs, numcep = 20, highfreq=Fs/2, winfunc = np.hamming, nfft = 1280)
    red = []
    for j in range (0,20):
        sum = 0
        for i in range(0,498):
            sum += coeffs[i][j]
        red.append(sum)


    features.append(red)
    cnt += 1
    print(cnt)


for genre in genres:
    n_gen = np.zeros(6)
    if (genre == 'classical'):
        n_gen[0] = 1
    elif (genre == 'folk'):
        n_gen[1] = 1
    elif (genre == 'house'):
        n_gen[2] = 1
    elif (genre == 'jazz'):
        n_gen[3] = 1
    elif (genre == 'rnb'):
        n_gen[4] = 1
    else:
        n_gen[5] = 1
    gen.append(n_gen)

with open("MFFC_BuiltInPy_Bane/MFCC_feature_gtzan.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')

    for red in features:
        writer.writerow(red)