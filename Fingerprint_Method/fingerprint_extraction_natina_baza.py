import os, csv
from numpy.core.fromnumeric import shape
import pydub
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav


final = []
with open('nata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line_cnt = 0
    
    # Reading every row on 'data.csv'
    for row in csv_reader:
        line_cnt += 1
        if (line_cnt == 1):
            continue

        song_path = row[0]
        genre = row[1]

        # Reading audio and preformind FFT on audio
        rate, data = scipy.io.wavfile.read(song_path)

        fourier = np.fft.fft(data)
        fourier = abs(fourier)

        # Extracting features:
        # 30-40 / 41-80 / 81-120 / 121-180 / 181-300
        fingerprint = np.array([
            np.argmax(fourier[29:40]) / 10,
            np.argmax(fourier[40:80]) / 40,
            np.argmax(fourier[80:120]) / 40,
            np.argmax(fourier[120:180]) / 60,
            np.argmax(fourier[180:300]) / 120,
            np.argmax(fourier[300:500]) / 200,
            np.argmax(fourier[500:700]) / 200,
            np.argmax(fourier[700:900]) / 200,
            np.argmax(fourier[900:1200]) / 300,
            np.argmax(fourier[1200:1500]) / 300,
            np.argmax(fourier[1800:2000]) / 200
        ])

        # Making genre + features string for .csv
        red = []
        red.append(genre)
        red.append(len(fingerprint))
        for x in fingerprint:
            red.append(str(x))

        final.append(red)

final = sorted(final)

# Inserting genre+features in .csv
header = np.array(['genre','number of features', 'features respectively'])  
with open("Fingerprint_Method/branko.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) # write the header

    for red in final:
        writer.writerow(red)
# classical, folk, house, jazz, rnb, rock