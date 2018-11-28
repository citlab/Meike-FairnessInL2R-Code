import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from learning import topp_prot
from learning import find
from learning import exposure
from learning import topp


class DELTR_Trainer():
    '''
    Disparate Exposure in Learning To Rank
    --------------------------------------

    A supervised learning to rank algorithm that incorporates a measure of performance and a measure
    of disparate exposure into its loss function. Trains a linear model based on performance and
    fairness for a protected group.
    By reducing disparate exposure for the protected group, increases the overall group visibility in
    the resulting rankings and thus prevents systematic biases against a protected group in the model,
    even though such bias might be present in the training data.

    At this point, only works for one protected group.


    :field __pathToTrainingData: file path to training data. Requires first column to contain the
                                 query ids and last column to contain the training judgments
                                 (in descending order, i.e. higher scores are better)
    :field __pathToModelFile: file path in which resulting model (i.e. list of feature weights) are stored
    :field __resultDir: directory where to save plots

    :field __numberIterations: iterations in gradient descent
    :field __learningRate: learning rate in gradient descent
    :field __initVar: range of values for initialization of weights
    :field __lambda: regularization constant

    :field __gamma: float parameter tuning the disparate exposure metric
    :field __protectedColumn: index of column in data that contains protected attribute
    :field __protectedAttribute: int that describes the protected feature in the training data
    '''

    def __init__(self, pathToTrainingData, pathToModelFile, resultDir, gamma, numIter, learningRate,
                 protCol, protAttr, initVar, lambdaa):
        # paths
        self.__pathToTrainingData = pathToTrainingData
        self.__pathToModelFile = pathToModelFile
        self.__resultDir = resultDir

        # constants for gradient descent
        self.__numberIterations = numIter
        self.__learningRate = learningRate
        self.__initVar = initVar
        self.__lambda = lambdaa

        # constants for DELTR
        self.__gamma = gamma
        self.__protectedColumn = protCol
        self.__protectedAttribute = protAttr

        # switch off exposure calculation if gamma is zero, because it's faster
        if gamma == 0:
            self.__noExposure = True

    def train(self, colorblind=False):
        '''
        TODO: write doc
        '''
        # read testfile and load training dataset
        data = pd.read_csv(self.__pathToTrainingData, decimal=',', header=None)
        if colorblind:
            # if training should be done without protected attribute, set whole protected column to zero
            data[self.__protectedColumn].values[:] = 0
            # also switch off exposure calculation, as not needed, which speeds up training
            self.__noExposure = True
        # TODO: Steffi --> warum wandelst du den dataframe in ein np-array um? schreib mal einen Kommentar dazu
        data = data.apply(pd.to_numeric, errors='ignore')
        print(data)

        # prepare data
        query_ids = np.asarray(data.iloc[:, 0])
        featureMatrix = np.asarray(data.iloc[:, 1:(data.shape[1] - 1)])
        trainingScores = np.reshape(np.asarray(data.iloc[:, data.shape[1] - 1]), (featureMatrix.shape[0], 1))

        # launch the training routine
        start = datetime.datetime.now()
        omega = self._trainNN(query_ids,
                              featureMatrix,
                              trainingScores)
        end = datetime.datetime.now()
        print("time: ", end - start)
        print("omega: ", omega)

        # safe results
        np.save(self.__pathToModelFile, omega)

    def _trainNN(self, query_ids, featureMatrix, trainingScores, quiet=False):
        """
        trains the Neural Network to find the optimal loss in listwise learning to rank

        :param query_ids: list of query IDs
        :param featureMatrix: training features
        :param trainingScores: training judgments
        :param quiet:
        """
        m = featureMatrix.shape[0]
        n_features = featureMatrix.shape[1]

        prot_idx = np.reshape(featureMatrix[:, self.__protCol] == np.repeat(self.__protAttr, m), (m, 1))
        # linear neural network parameter initialization
        omega = (np.random.rand(n_features, 1) * self.__initVar).reshape(-1)

        cost_converge_J = np.zeros((self.__numberIterations, 1))
#         cost_converge_L = np.zeros((self.__numberIterations, 1))
#         cost_converge_U = np.zeros((self.__numberIterations, 1))
        omega_converge = np.empty((self.__numberIterations, n_features))

        # training routine
        for t in range(0, self.__numberIterations):
            if quiet == False:
                print('iteration ', t)

            # forward propagation
            predictedScores = np.dot(featureMatrix, omega)
            predictedScores = np.reshape(np.asarray(predictedScores).astype('float'), (len(predictedScores), 1))
            # cost
            if quiet == False:
                print('computing cost')

            # with regularization
            cost = self._calculateCost(trainingScores, predictedScores, query_ids, prot_idx)
            J = cost + np.transpose(np.multiply(predictedScores, predictedScores)) * self.__lambda
            cost_converge_J[t] = np.sum(J)

            if quiet == False:
                print("computing gradient")

            grad = self._calculateGradient(featureMatrix, trainingScores, predictedScores, query_ids, prot_idx)
            omega = omega - self.__learningRate * np.sum(np.asarray(grad)[0], axis=1)
            omega_converge[t, :] = np.transpose(omega[:])

            if quiet == False:
                print('\n')

        # plots
        plt.subplot(211)
        plt.plot(cost_converge_J)
        plt.subplot(212)
        plt.plot(omega_converge)
        plt.savefig(self.__resultDir + 'cost_and_gradient.png')

        return omega

    def _calculateCost(self, training_judgments, predictions, query_ids, prot_idx):
        """
        computes the loss in list-wise learning to rank
        it incorporates L which is the error between the training judgments and those
        predicted by a model and U which is the disparate exposure metric
        implementation of equation 6 in DELTR paper

        :param training_judgments: containing the training judgments/ scores
        :param predictions: containing the predicted scores
        :param query_ids: list of query IDs
        :param prot_idx: list stating which item is protected or non-protected
        :return: a float value --> loss
        """
        data_per_query = lambda which_query, data: \
                                       find.find_items_per_group_per_query(data,
                                                                           query_ids,
                                                                           which_query,
                                                                           prot_idx)

        # eq 2 from DELTR paper
        loss = lambda which_query: \
                            -np.dot(np.transpose(topp.topp(data_per_query(which_query,
                                                                          training_judgments)[0])),
                                    np.log(topp.topp(data_per_query(which_query, predictions)[0])))

        if self.__noExposure:
            cost = lambda which_query: loss(which_query)
        else:
            # eq 6 from DELTR paper
            cost = lambda which_query:  self.__gamma \
                                        * exposure.exposure_diff(predictions,
                                                                 query_ids,
                                                                 which_query,
                                                                 prot_idx) \
                                        ** 2 \
                                        +loss(which_query)

        results = [cost(query) for query in query_ids]

        return np.asarray(results)

    def _calculateGradient(self, training_features, training_judgments, predictions, query_ids, prot_idx):
        """
        finds the optimal solution for the listwise cost
        implementation of equation 8 and appendix A in paper DELTR

        :param training_features: containing all the features
        :param training_judgments: vector containing the training judgments/ scores
        :param predictions: vector containing the prediction scores
        :param query_ids: list of query IDs
        :param prot_idx: list stating which item is protected or non-protected
        :return: float value --> optimal listwise cost
        """
        # find all training judgments and all predicted scores that belong to one query
        data_per_query = lambda which_query, data: \
                                       find.find_items_per_group_per_query(data,
                                                                           query_ids,
                                                                           which_query,
                                                                           prot_idx)
        # Exposure in rankings for protected and non-protected group, right summand in eq 8
        U_deriv = lambda which_query:   2 \
                                        * exposure.exposure_diff(predictions,
                                                                 query_ids,
                                                                 which_query,
                                                                 prot_idx) \
                                        * topp_prot.normalized_topp_prot_deriv_per_group_diff(training_features,
                                                                                              predictions,
                                                                                              query_ids,
                                                                                              which_query,
                                                                                              prot_idx)
        # Training error
        l1 = lambda which_query: np.dot(np.transpose(data_per_query(which_query,
                                                                    training_features)[0]),
                                        topp.topp(data_per_query(which_query,
                                                                 training_judgments)[0]))
        l2 = lambda which_query: 1 \
                                 / np.sum(np.exp(data_per_query(which_query,
                                                                predictions)[0]))
        l3 = lambda which_query: np.dot(np.transpose(data_per_query(which_query,
                                                                    training_features)[0]),
                                        np.exp(data_per_query(which_query,
                                                              predictions)[0]))

        L_deriv = lambda which_query:-l1(which_query) + l2(which_query) * l3(which_query)

        if self.__noExposure:
            # standard L2R that only considers loss
            grad = lambda which_query: L_deriv(which_query)
        else:
            # eq 8 in DELTR paper
            grad = lambda which_query: self.__gamma * U_deriv(which_query) + L_deriv(which_query)

#         if Globals.ONLY_U:
#             grad = lambda which_query: gamma * U_deriv(which_query)

        results = [grad(query) for query in query_ids]

        return np.asarray(results)

