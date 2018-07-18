'''
Created on Jun 28, 2018

@author: mzehlike
'''
import unittest
import numpy as np
from src.learning import find
from numpy import dtype


class TestFindMethods(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.__query_ids = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3, 4])
        self.__data = np.matrix('1 0 10; 1 1 9; 1 1 8; 1 0 7; 2 0 6; 2 1 5; 3 0 4; 3 0 3; 3 1 2; 4 1 1')


    @classmethod
    def tearDownClass(cls):
        super(TestFindMethods, cls).tearDownClass()


    def test_find_items_per_query(self):
        expected_1 = np.matrix('1 0 10; 1 1 9; 1 1 8; 1 0 7')
        actual_1 = find.find_items_per_query(self.__data, self.__query_ids, 1)
        np.testing.assert_array_equal(expected_1, actual_1)

        expected_2 = np.matrix('2 0 6; 2 1 5')
        actual_2 = find.find_items_per_query(self.__data, self.__query_ids, 2)
        np.testing.assert_array_equal(expected_2, actual_2)

        expected_3 = np.matrix('3 0 4; 3 0 3; 3 1 2')
        actual_3 = find.find_items_per_query(self.__data, self.__query_ids, 3)
        np.testing.assert_array_equal(expected_3, actual_3)

        expected_4 = np.matrix('4 1 1')
        actual_4 = find.find_items_per_query(self.__data, self.__query_ids, 4)
        np.testing.assert_array_equal(expected_4, actual_4)

        expected_5 = np.matrix('', dtype='int64')
        actual_5 = find.find_items_per_query(self.__data, self.__query_ids, 5)
        self.assertEqual(expected_5.size, actual_5.size)


    def test_find_items_per_group_per_query(self):
        self.fail("TODO STEFFI: not yet implemented")





