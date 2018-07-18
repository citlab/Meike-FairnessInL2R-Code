'''
Created on Jun 23, 2018

@author: mzehlike
'''
import unittest
import numpy as np
from learning.topp_prot import topp_prot

class Test_ToppProt(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_toppProt_standard(self):
        all_items = np.array([10, 9, 2, 1])
        prot = np.array([10, 9])
        nprot = np.array([2, 1])

        expected_prot = np.exp(prot) / np.sum(np.exp(all_items))
        actual_prot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot) / np.sum(np.exp(all_items))
        actual_nprot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)


    def test_toppProt_OneGroupOnlyZeros(self):
        all_items = np.array([10, 9, 0, 0])
        prot = np.array([10, 9])
        nprot = np.array([0, 0])

        expected_prot = np.exp(prot) / np.sum(np.exp(all_items))
        actual_prot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot) / np.sum(np.exp(all_items))
        actual_nprot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)


    def test_toppProtFirstDerivative_standard(self):
        self.fail("not yet implemented")
        all_predictions = np.array([4, 3, 2, 1])
        prot_predictions = np.array([2, 1])
        nprot_predictions = np.array([4, 3])

        all_features = np.matrix('0 10; 0 9; 1 2; 1 1')
        prot_features = np.matrix('1 2; 1 1')
        nprot_features = np.matrix('0 10; 0 9')

        # calculated expected number by using eq 11 of CIKM paper
        expected_prot = np.exp(prot_predictions) / np.sum(np.exp(all_predictions))
        actual_prot = topp_prot(prot_predictions, all_predictions)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot_predictions) / np.sum(np.exp(all_predictions))
        actual_nprot = topp_prot(prot_predictions, all_predictions)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)


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
