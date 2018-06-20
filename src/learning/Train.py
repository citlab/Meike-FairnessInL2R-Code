import sys
import csv
import numpy as np
from TrainNN import *

#read arguments from the command line
arg_list = sys.argv
print('Argument List:', str(sys.argv))

directory = arg_list[1]
training_file = arg_list[2]
model_file = arg_list[3]
GAMMA = float(arg_list[4])

#read testfile and load training dataset
with open('testdaten.csv') as csvfile:
    readCSV = csv.reader(csvfile,delimiter=';')
    #row_count = sum(1 for row in readCSV)
    list_id = []
    y = []
    X = []
    for row in readCSV:
        id = row[0]
        X1 = row[1:len(row)-1]
        y1 = row[len(row)-1]

        list_id.append(id)
        X.append(X1)
        y.append(y1)
#print(str(X_list))
list_id = np.asarray(list_id)
X = np.asarray(X)
y = np.asarray(y)
print(y.shape[0])
#launch the training routine
#omega = trainNN(GAMMA, directory, list_id, X, y)



