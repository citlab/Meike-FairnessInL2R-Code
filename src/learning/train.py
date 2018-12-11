import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt


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
                 protCol, protAttr, initVar, lambdaa, quiet=False):
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
        self.__noExposure = False
        if gamma == 0:
            self.__noExposure = True

        # print intermediate steps to stdout
        self.__quiet = quiet

    def train(self, colorblind=False):
        '''
        prepares data and launches training routine.
        also measures time for the calculation and saves resulting model to disk
        '''
        # read testfile and load training dataset
        data = pd.read_csv(self.__pathToTrainingData, decimal=',', header=None)
        if colorblind:
            # if training should be done without protected attribute, set whole protected column to zero
            data[self.__protectedColumn].values[:] = 0
            # also switch off exposure calculation, as not needed, which speeds up training
            self.__noExposure = True
        # TODO: Steffi --> warum wandelst du den dataframe in ein np-array um? schreib mal einen Kommentar dazu
        # cast dataframe entries to numeric values
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

    def _trainNN(self, query_ids, featureMatrix, trainingScores):
        """
        trains the Neural Network to find the optimal feature weights in listwise learning to rank

        :param query_ids: list of query IDs
        :param featureMatrix: training features
        :param trainingScores: training judgments
        """
        m = featureMatrix.shape[0]
        n_features = featureMatrix.shape[1]

        prot_idx = np.reshape(featureMatrix[:, self.__protectedColumn] == np.repeat(self.__protectedAttribute, m), (m, 1))
        # linear neural network parameter initialization
        omega = (np.random.rand(n_features, 1) * self.__initVar).reshape(-1)
        #omega = (0.01, 0.01)

        cost_converge_J = np.zeros((self.__numberIterations, 1))
#         cost_converge_L = np.zeros((self.__numberIterations, 1))
#         cost_converge_U = np.zeros((self.__numberIterations, 1))
        omega_converge = np.empty((self.__numberIterations, n_features))

        # training routine
        for t in range(0, self.__numberIterations):
            if self.__quiet == False:
                print('iteration ', t)

            # forward propagation
            predictedScores = np.dot(featureMatrix, omega)
            predictedScores = np.reshape(predictedScores,(featureMatrix.shape[0], 1))
            #predictedScores = np.reshape(np.asarray(predictedScores).astype('float'), (len(predictedScores), 1))

            # cost
            if self.__quiet == False:
                print('computing cost')

            # with regularization
            cost = self._calculateCost(trainingScores, predictedScores, query_ids, prot_idx)
            J = cost + np.transpose(np.multiply(predictedScores, predictedScores)) * self.__lambda
            cost_converge_J[t] = np.sum(J)

            if self.__quiet == False:
                print("computing gradient")

            grad = self._calculateGradient(featureMatrix, trainingScores, predictedScores, query_ids, prot_idx)
            omega = omega - self.__learningRate * np.sum(np.asarray(grad), axis=0).reshape(-1)
            omega_converge[t, :] = np.transpose(omega[:])

            if self.__quiet == False:
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
                                       self._find_items_per_group_per_query(data,
                                                                             query_ids,
                                                                             which_query,
                                                                             prot_idx)

        # eq 2 from DELTR paper
        loss = lambda which_query: \
                            -np.dot(np.transpose(self._topp(data_per_query(which_query,
                                                                           training_judgments)[0])),
                                    np.log(self._topp(data_per_query(which_query,
                                                                     predictions)[0])))/np.log(predictions.size)

        if self.__noExposure:
            cost = lambda which_query: loss(which_query)
        else:
            # eq 6 from DELTR paper
            cost = lambda which_query:  self.__gamma \
                                        * self._exposure_diff(predictions,
                                                              query_ids,
                                                              which_query,
                                                              prot_idx) \
                                        ** 2 \
                                        +loss(which_query)
        print("U: {}".format(self._exposure_diff(predictions,query_ids,1,prot_idx)))
        print("L: {}".format(loss(1)))

        results = [cost(query) for query in query_ids]

        return np.asarray(results)

    def _calculateGradient(self, training_features, training_judgments, predictions, query_ids, prot_idx):
        """
        calculates local gradients of current feature weights
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
                                       self._find_items_per_group_per_query(data,
                                                                            query_ids,
                                                                            which_query,
                                                                            prot_idx)
        # Exposure in rankings for protected and non-protected group, right summand in eq 8
        U_deriv = lambda which_query:   2 \
                                        * self._exposure_diff(predictions,
                                                              query_ids,
                                                              which_query,
                                                              prot_idx) \
                                        * self._normalized_topp_prot_deriv_per_group_diff(training_features,
                                                                                          predictions,
                                                                                          query_ids,
                                                                                          which_query,
                                                                                          prot_idx)
        # Training error
        l1 = lambda which_query: np.dot(np.transpose(data_per_query(which_query,
                                                                    training_features)[0]),
                                        self._topp(data_per_query(which_query,
                                                                  training_judgments)[0]))
        l2 = lambda which_query: 1 \
                                 / np.sum(np.exp(data_per_query(which_query,
                                                                predictions)[0]))
        l3 = lambda which_query: np.dot(np.transpose(data_per_query(which_query,
                                                                    training_features)[0]),
                                        np.exp(data_per_query(which_query,
                                                              predictions)[0]))

        L_deriv = lambda which_query:(-l1(which_query) + l2(which_query) * l3(which_query))/np.log(predictions.size)

        if self.__noExposure:
            # standard L2R that only considers loss
            grad = lambda which_query: L_deriv(which_query)
        else:
            # eq 8 in DELTR paper
            grad = lambda which_query: self.__gamma * U_deriv(which_query) + L_deriv(which_query).reshape(-1)

#         if Globals.ONLY_U:
#             grad = lambda which_query: gamma * U_deriv(which_query)
        results = [grad(query) for query in query_ids]
        return np.asarray(results)

    def _find_items_per_group_per_query(self, data, query_ids, which_query, prot_idx):
        """
        finds all the items with a given query ID and separates the items into protected
        and non-protected groups

        :param data: all predictions or training judgments
        :param query_ids: list of query IDs
        :param which_query: given query ID
        :param prot_idx: list stating which item is protected or non-protected
        :return: three matrices
        """
        judgments_per_query = self.__find_items_per_query(data,
                                                          query_ids,
                                                          which_query)
        prot_idx_per_query = self.__find_items_per_query(prot_idx,
                                                         query_ids,
                                                         which_query)
        protected_items_per_query = judgments_per_query[np.where(prot_idx_per_query == True)[0], :]
        nonprotected_items_per_query = judgments_per_query[np.where(prot_idx_per_query == False)[0], :]

        return judgments_per_query, protected_items_per_query, nonprotected_items_per_query

    def __find_items_per_query(self, data, query_ids, which_query):
        """
        finds items which contains the given query_id

        :param data: all predictions or training judgments
        :param query_ids: list of query IDs
        :param which_query: given query ID
        :return: matrix filtered by which_query
        """
        return data[np.where(query_ids == which_query)[0], :]

    def _exposure_diff(self, data, query_ids, which_query, prot_idx):
        """
        computes the exposure difference between protected and non-protected groups
        implementation of equation 5 in DELTR paper but without the square

        :param data: all predictions
        :param query_ids: list of query IDs
        :param which_query: given query ID
        :param prot_idx: list states which item is protected or non-protected

        :return: float value
        """
        judgments_per_query, \
        protected_items_per_query, \
        nonprotected_items_per_query = self._find_items_per_group_per_query(data,
                                                                             query_ids,
                                                                             which_query,
                                                                             prot_idx)
        exposure_prot = self.__normalized_exposure(protected_items_per_query,
                                                   judgments_per_query)
        exposure_nprot = self.__normalized_exposure(nonprotected_items_per_query,
                                                    judgments_per_query)
        exposure_diff = np.maximum(0, (exposure_nprot - exposure_prot))

        return exposure_diff

    def __normalized_exposure(self, group_data, all_data):
        '''
        calculates the exposure of a group in the entire ranking
        implementation of equation 4 in DELTR paper

        :param group_data: predictions of relevance scores for one group
        :param all_data: all predictions

        :return: float value that is normalized exposure in a ranking for one group
                 nan if group size is 0
        '''
        return (np.sum(self.__topp_prot(group_data, all_data) / np.log(2))) / group_data.size

    def __topp_prot(self, group_items, all_items):
        '''
        given a dataset of features what is the probability of being at the top position
        for one group (group_items) out of all items
        example: what is the probability of each female (or male respectively) item (group_items) to be in position 1
        implementation of equation 7 in paper DELTR

        :param group_items: vector of predicted scores of one group (protected or non-protected)
        :param all_items: vector of predicted scores of all items

        :return: numpy array of float values
        '''
        return np.exp(group_items) / np.sum(np.exp(all_items))

    def _normalized_topp_prot_deriv_per_group_diff(self, training_features, predictions, query_ids, which_query, prot_idx):
        """
        calculates the difference of the normalized topp_prot derivative of the protected and non-protected groups
        implementation of the second factor equation 12 in paper DELTR

        :param training_features: vector of all features
        :param predictions: predictions of all data points
        :param query_ids: list of query IDs
        :param which_query: given query
        :param prot_idx: list stating which item is protected or non-protected

        :return: numpy array of float values
        """
        train_judgments_per_query, \
        train_protected_items_per_query, \
        train_nonprotected_items_per_query = self._find_items_per_group_per_query(training_features,
                                                                                   query_ids,
                                                                                   which_query,
                                                                                   prot_idx)

        predictions_per_query, \
        pred_protected_items_per_query, \
        pred_nonprotected_items_per_query = self._find_items_per_group_per_query(predictions,
                                                                                  query_ids,
                                                                                  which_query,
                                                                                  prot_idx)

        u2 = self.__normalized_topp_prot_deriv_per_group(\
            train_nonprotected_items_per_query, \
            train_judgments_per_query, \
            pred_nonprotected_items_per_query, \
            predictions_per_query)  # derivative for non-protected group
        u3 = self.__normalized_topp_prot_deriv_per_group(\
            train_protected_items_per_query, \
            train_judgments_per_query, \
            pred_protected_items_per_query, \
            predictions_per_query)  # derivative for protected group

        return u2 - u3

    def __normalized_topp_prot_deriv_per_group(self, group_features, all_features, group_predictions, all_predictions):
        """
        normalizes the results of the derivative of topp_prot

        :param group_features: feature vector of (non-) protected group
        :param all_features: feature vectors of all data points
        :param group_predictions: predictions of all data points
        :param all_predictions: predictions of all data points

        :return: numpy array of float values
        """
        derivative = self.__topp_prot_first_derivative(group_features,
                                                       all_features,
                                                       group_predictions,
                                                       all_predictions)
        result = (np.sum(derivative / np.log(2), axis=0)) / group_predictions.size
        return result

    def __topp_prot_first_derivative(self, group_features, all_features, group_predictions, all_predictions):
        '''
        Derivative for topp_prot in pieces:
        implementation of equation 11 in paper DELTR

        :param group_features: feature vector of (non-) protected group
        :param group_predictions: predicted scores for (non-) protected group
        :param all_predictions: predictions of all data points
        :param all_features: feature vectors of all data points

        :return: numpy array with weight adjustments
        '''
        # numerator1 = np.dot(group_features, np.repeat(np.exp(group_predictions), group_features.shape[0]))
        numerator1 = np.dot(np.transpose(np.exp(group_predictions)),
                            group_features)
        numerator2 = np.sum(np.exp(all_predictions))
        numerator3 = np.sum(np.dot(np.transpose(np.exp(all_predictions)),
                                   all_features))
        denominator = np.sum(np.exp(all_predictions)) ** 2

        result = (numerator1 * numerator2 - np.exp(group_predictions) * numerator3) / denominator

        # return result as flat numpy array instead of matrix
        return np.asarray(result)

    def _topp(self, v):
        """
        computes the probability of a document being
        in the first position of the ranking
        implementation of equation 7 in paper DELTR

        :param v: all training judgments or all predictions
        :return: float value which is a probability
        """
        return np.exp(v) / np.sum(np.exp(v))
