'''
Created on Jun 23, 2018

@author: mzehlike
'''
import unittest
import numpy as np
from src.learning.topp_prot import topp_prot
from src.learning.topp_prot import topp_prot_first_derivative

class Test_ToppProt(unittest.TestCase):


    def test_toppProt_standard(self):
        all_items = np.array([10, 9, 2, 1])
        prot = np.array([10, 9])
        nprot = np.array([2, 1])

        expected_prot = np.exp(prot) / np.sum(np.exp(all_items))
        actual_prot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot) / np.sum(np.exp(all_items))
        actual_nprot = topp_prot(nprot, all_items)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)


    def test_toppProt_OneGroupOnlyZeros(self):
        all_items = np.array([10, 9, 0, 0])
        prot = np.array([10, 9])
        nprot = np.array([0, 0])

        expected_prot = np.exp(prot) / np.sum(np.exp(all_items))
        actual_prot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot) / np.sum(np.exp(all_items))
        actual_nprot = topp_prot(nprot, all_items)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)


    def test_toppProtFirstDerivative_standard(self):
        all_predictions = np.array([4, 3, 2, 1])
        prot_predictions = np.array([2, 1])
        nprot_predictions = np.array([4, 3])

        all_features = np.matrix('0 10; 0 9; 1 2; 1 1')
        prot_features = np.matrix('1 2; 1 1')
        nprot_features = np.matrix('0 10; 0 9')

        # calculated expected number by using eq 11 of CIKM paper, see text file
        # "calculateDerivativeByHand" in test folder
        expected_prot = np.array([-0.656089, -0.078867])
        actual_prot = topp_prot_first_derivative(prot_features, all_features, prot_predictions, all_predictions)
        np.testing.assert_array_almost_equal(expected_prot, actual_prot)

        expected_nprot = np.array([-5.728672, 6.463627])
        actual_nprot = topp_prot_first_derivative(nprot_features, all_features, nprot_predictions, all_predictions)
        np.testing.assert_array_almost_equal(expected_nprot, actual_nprot)


    def test_toppProtFirstDerivative_OneGroupOnlyZeros(self):
        self.fail("not yet implemented")
        all_items = np.array([10, 9, 0, 0])
        prot = np.array([10, 9])
        nprot = np.array([0, 0])

        expected_prot = np.exp(prot) / np.sum(np.exp(all_items))
        actual_prot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot) / np.sum(np.exp(all_items))
        actual_nprot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)

