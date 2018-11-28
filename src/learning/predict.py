import numpy as np
import pandas as pd


class DELTR_Predictor():
    '''
    :field __pathToTestData: string with path to test data file
    :field __pathToModelFile: string with path to model file, which contains feature weights
    :field __resultDir: string with directory into which results are stored
    :field __protectedColumn: int with index of column that contains protected attribute in test data
    :field __quiet: if True, no command line outputs are printed during calculation
    '''

    def __init__(self, pathToTestData , pathToModelFile, resultDir, protCol, quiet=False):
        self.__pathToTestData = pathToTestData
        self.__pathToModelFile = pathToModelFile
        self.__resultDir = resultDir

        self.__protectedColumn = protCol
        self.__quiet = quiet

    def predict(self):
        testData = pd.read_csv(self.__pathToTestData, decimal=',')
        testData = testData.apply(pd.to_numeric, errors='ignore')
        if self.__quiet == False:
            print(testData)

        omega = self._getOmega()
        query_ids, features, trainingScores = self._prepareData(testData)
        predictions = np.dot(features, omega)
        doc_ids = np.arange(1, np.size(predictions) + 1)

        # also write trainingScores for later evaluation
        trainingScores = np.transpose(np.array([query_ids, doc_ids,
                                                trainingScores.reshape(-1),
                                                features[:, self.__protectedColumn]]))
        np.savetxt(self.__resultDir + 'trainingScores_ORIG.pred', trainingScores, delimiter=',', fmt=['%d', '%d', '%1.2f', '%d'])

        # unsorted prediction
        predictions_unsorted = np.transpose(np.array([query_ids, doc_ids, predictions, features[:, self.__protectedColumn]]))
        np.savetxt(self.__resultDir + 'predictions_UNSORTED.pred', predictions_unsorted, delimiter=',', fmt=['%d', '%d', '%f', '%d'])

        # sorted prediction
        predictions_sorted = predictions.argsort()[::-1]
        predictions_sorted = np.transpose(np.array([query_ids[predictions_sorted],
                                                    doc_ids[predictions_sorted],
                                                    predictions[predictions_sorted],
                                                    features[:, self.__protectedColumn][predictions_sorted]]))
        np.savetxt(self.__resultDir + 'predictions_SORTED.pred', predictions_sorted, delimiter=',', fmt=['%d', '%d', '%f', '%d'])

    def _getOmega(self):
        '''
        read model from disk and return it as np-array containing floats
        '''
        with open(self.__pathToModelFile, 'r', encoding='utf-8-sig') as file:
            for line in file:
                omega = np.asarray(line.split(" "), dtype='float64')
                if self.__quiet == False:
                    print(omega)
        return omega

    def _prepareData(self, testData):
        '''
        splits given dataframe into three numpy arrays, containing query ids, features and scores
        '''
        query_ids = np.asarray(testData.iloc[:, 0])
        features = np.asarray(testData.iloc[:, 1:(testData.shape[1] - 1)])
        trainingScores = np.reshape(np.asarray(testData.iloc[:, testData.shape[1] - 1]), (features.shape[0], 1))
        return query_ids, features, trainingScores

