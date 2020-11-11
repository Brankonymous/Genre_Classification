import numpy as np 
import os
import csv

files_path = 'database/'
labels = np.array([])
paths = np.array([])

for root, dirs, files in os.walk(files_path):
    for file in files:
        s = file[7:]
        broj = ''
        for i in s:
            if i == ')':
                break
            broj+=i

        broj = int(broj)        

        
        if broj<300:
            zanr = 'classical'
        elif broj<600:
            zanr = 'folk'
        elif broj<900:
            zanr = 'house'
        elif broj<1200:
            zanr = 'jazz'
        elif broj<1500:
            zanr = 'rnb'
        else:
            zanr = 'rock'
        
        labels = np.append(labels, zanr)
        paths = np.append(paths, files_path + file)

final = np.concatenate((np.reshape(paths,(len(paths),1) ), np.reshape(labels,(len(labels),1) )),axis = 1)

header = np.array(['path','genre'])

with open("nata.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) 
    for l in final:
        writer.writerow(l)

