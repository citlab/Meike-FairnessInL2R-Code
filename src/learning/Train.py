import sys
import csv
import numpy as np
from src.learning import TrainNN
from src.learning import Globals

#read arguments from the command line
arg_list = sys.argv
##print('Argument List:', str(sys.argv))

#directory = arg_list[1]
#training_file = arg_list[2]
#model_file = arg_list[3]
#GAMMA = float(arg_list[4])
GAMMA = 0
directory = 'C:\\Users\\ying_\\Pictures\\Listwise_Cost'

#read testfile and load training dataset
with open('testdaten.csv',encoding='utf-8-sig') as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
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
list_id = np.asarray(list_id).astype('int64')
X = np.asarray(X).astype('float')
y = np.reshape(np.asarray(y).astype('float'),(len(y),1))

#launch the training routine
omega = TrainNN.trainNN(GAMMA, directory, list_id, X, y, Globals.T, Globals.e)



