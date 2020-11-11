import pydub
import os
import numpy as np
from pydub import AudioSegment
from pydub.audio_segment import extract_wav_headers
import csv

files_path = 'genres/bigger_cut'

paths = np.array([])
labels = np.array([])

for root, dirs, files in os.walk(files_path):
    for file in files:
        s = ''
        for i in file:
            if (i == '.' or i == '_'):
                break
            s+=i
        
        tempPath = root + '/' + file
        paths = np.append(paths, tempPath)
        labels = np.append(labels, s)

final = np.concatenate((np.reshape(paths,(len(paths),1) ), np.reshape(labels,(len(labels),1) )),axis = 1)

header = np.array(['path','genre'])

with open("updated_gtzan.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) 
    for l in final:
        writer.writerow(l)
