'''
Created on Jun 28, 2018

@author: mzehlike
'''
import unittest
import numpy as np


class TestFindMethods(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.__query_ids = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3, 4])
        self.__protection_status = np.matrix('0 10; 1 9; 1 8; 0 7; 0 6; 1 5; 0 4; 0 3; 1 2; 1 1')


    @classmethod
    def tearDownClass(cls):
        super(TestFindMethods, cls).tearDownClass()


    def test_find_items_per_query(self):
        expected_1 = np.matrix('0 10; 1 9; 1 8; 0 7')
        pass


