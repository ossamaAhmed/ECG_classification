import numpy as np
import pandas as pd
import biosppy as bio
import time
#Input file names
x_file="X_train.csv"
y_file="y_train.csv"
x_test_file="X_test.csv"

#Import csv files as data frames
y_train = pd.read_csv(y_file, header=0)
N=y_train.size
m=180
x_train=np.zeros((N,2*m))
i=0
#Extract, from the data frames, the 2D arrays of values
t = time.time()

import csv
with open(x_file, newline='') as f:
    reader = csv.reader(f)
    first=True
    for row in reader:
        if first:
            first=False
            continue
        #print('time0=', time.time() - t)
        #t = time.time()
        sig=np.array(row[1:])
        ecg=bio.ecg.ecg(signal=sig.astype(int), sampling_rate=300, show=False)
        #print('time2=', time.time() - t)
        #t = time.time()
        beats=ecg['templates']
        mean_beat=np.mean(beats,0)
        var_beat=np.var(beats,0)
        x_train[i,:]=np.concatenate((mean_beat,var_beat))
        i+=1
        if i % 100==0:
            print(i/N)
np.savetxt("X_train_extracted.csv", x_train, delimiter=",")

#Extracting features for test data
x_test=[]
indeces=[]
i=0
print('Done Train')
with open(x_test_file) as f:
    reader = csv.reader(f)
    first=True
    for row in reader:
        if first:
            first=False
            continue
        sig=np.array(row[1:])
        ecg=bio.ecg.ecg(signal=sig.astype(int), sampling_rate=300, show=False)

        beats=ecg['templates']
        heart_rate=ecg['heart_rate']
        mean_beat=np.mean(beats,0)
        var_beat=np.var(beats,0)
        x_test.append(np.concatenate((mean_beat,var_beat)))
        i+=1
        indeces.append(i)

np.savetxt("X_test_extracted.csv", x_test, delimiter=",")