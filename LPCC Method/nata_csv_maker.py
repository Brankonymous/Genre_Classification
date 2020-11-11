import csv


with open("LPCC Method/nata.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    
    for i in range(0,1800):
        s = 'database/pesma ('
        q = ').wav'
        final = s + str(i+1) + q
        red = []
        red.append(final)
        if (i<300):
            red.append('classical')
        elif (i<600):
            red.append('folk')
        elif (i<900):
            red.append('house')
        elif (i<1200):
            red.append('jazz')
        elif (i<1500):
            red.append('rnb')
        else:
            red.append('rock')

        writer.writerow(red)