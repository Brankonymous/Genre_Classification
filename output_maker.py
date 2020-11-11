import csv
import numpy as np


with open("Combining_Features/nata.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    
    red = []
    for i in range(0,1800):
        x = int (i / 300)
        red = np.zeros(6)
        red[x] = 1
        
        writer.writerow(red)
