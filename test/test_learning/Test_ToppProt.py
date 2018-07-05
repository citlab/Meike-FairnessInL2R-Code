'''
Created on Jun 23, 2018

@author: mzehlike
'''
import unittest
import numpy as np
from src.learning.topp_prot import topp_prot

class Test_ToppProt(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testOldVersion(self):
        all_items = np.array([10, 9, 2, 1])
        prot = np.array([10, 9])
        nprot = np.array([2, 1])

        expected_prot = np.exp(prot) / np.sum(np.exp(all_items))
        actual_prot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot) / np.sum(np.exp(all_items))
        actual_nprot = topp_prot(nprot, all_items)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)


    def testOneGroupOnlyZeros(self):
        all_items = np.array([10, 9, 0, 0])
        prot = np.array([10, 9])
        nprot = np.array([0, 0])

        expected_prot = np.exp(prot) / np.sum(np.exp(all_items))
        actual_prot = topp_prot(prot, all_items)
        np.testing.assert_array_equal(expected_prot, actual_prot)

        expected_nprot = np.exp(nprot) / np.sum(np.exp(all_items))
        actual_nprot = topp_prot(nprot, all_items)
        np.testing.assert_array_equal(expected_nprot, actual_nprot)


    def testNewImplementationAfterFixme(self):
        self.fail("not yet implemented")

    def testToppProtFirstDerivative(self):
        all_features = np.array([1,3,4,5,6,2])
