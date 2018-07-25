import unittest
import numpy as np
from learning import exposure
from learning import topp_prot

class TestExposureMethod(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.__data = np.matrix('10;9; 8; 7')
        self.__prot_group = np.matrix('9; 8')
        self.__nprot_group = np.matrix('10; 7')

    def tearDown(self):
        pass

    def test_exposure_for_each_group(self):
        expected_prot_1 = np.sum(topp_prot.topp_prot(self.__prot_group, self.__data) / np.log(2)) / self.__prot_group.size
        actual_prot_1 = exposure.normalized_exposure(self.__prot_group, self.__data)
        np.testing.assert_equal(expected_prot_1, actual_prot_1)

        expected_nprot_1 = np.sum(topp_prot.topp_prot(self.__nprot_group, self.__data) / np.log(2)) / self.__nprot_group.size
        actual_nprot_1 = exposure.normalized_exposure(self.__nprot_group, self.__data)
        np.testing.assert_equal(expected_nprot_1, actual_nprot_1)

    def test_exposure_for_empty_group(self):
        data = np.matrix('10; 9; 8; 7')
        prot_group = np.matrix('9; 8')
        nprot_group = np.matrix('')

        expected_prot_2 = np.sum(topp_prot.topp_prot(prot_group, data) / np.log(2)) / prot_group.size
        actual_prot_2 = exposure.normalized_exposure(prot_group, data)
        np.testing.assert_equal(expected_prot_2, actual_prot_2)

        expected_nprot_2 = np.sum(topp_prot.topp_prot(nprot_group, data) / np.log(2)) / nprot_group.size
        actual_nprot_2 = exposure.normalized_exposure(nprot_group, data)
        np.testing.assert_equal(expected_nprot_2, actual_nprot_2)




