'''
Created on Jun 27, 2018

@author: mzehlike
'''
import unittest
import numpy as np
from learning.topp import topp
from learning import exposure
from learning.Listwise_cost import listwise_cost

class Test(unittest.TestCase):


    def setUp(self):
        self.__GAMMA = 1
        self.__training_judgments = np.matrix('10; 9; 8; 7')
        self.__predictions = np.matrix('2; 3; 4; 5')
        self.__query_ids = np.array([1, 1, 1, 1])
        self.__prot_idx = np.matrix('False; True; True; False')


    def testListwiseCost(self):
        P_train = topp(self.__training_judgments)
        P_pred = topp(self.__predictions)
        L = -np.dot(np.transpose(P_train), np.log(P_pred))
        U = exposure.exposure_diff(self.__predictions, self.__query_ids, 1, self.__prot_idx)**2
        expected = L + self.__GAMMA * U
        actual = listwise_cost(self.__GAMMA, self.__training_judgments, self.__predictions, self.__query_ids, self.__prot_idx)[0]
        np.testing.assert_equal(expected, actual)


