import numpy as np
import datetime
import pandas as pd
from learning import TrainNN


# TODO: bash Skripte anpassen, sodass Experimente mit Python laufen
# TODO: refactoring sodass keine globalen Variablen mehr gibt, evtl object-oriented
def train(pathToTrainingData, pathToModelFile, resultDir, gamma, numIter, learningRate, protCol, protAttr):
    '''
    TODO: write doc
    '''
    # read testfile and load training dataset
    df = pd.read_csv(pathToTrainingData, decimal=',', header=None)
    df = df.apply(pd.to_numeric, errors='ignore')
    print(df)
    list_id = np.asarray(df.iloc[:, 0])
    featureMatrix = np.asarray(df.iloc[:, 1:(df.shape[1] - 1)])
    trainingScores = np.reshape(np.asarray(df.iloc[:, df.shape[1] - 1]), (featureMatrix.shape[0], 1))
    start = datetime.datetime.now()
    # launch the training routine
    omega = TrainNN.trainNN(gamma, resultDir, list_id, featureMatrix, trainingScores, numIter, learningRate, protCol, protAttr)
    end = datetime.datetime.now()
    print("time: ", end - start)
    print("omega: ", omega)
    np.save(pathToModelFile, omega)
