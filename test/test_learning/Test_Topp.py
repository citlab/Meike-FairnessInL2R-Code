'''
Created on Jun 23, 2018

@author: mzehlike
'''
import unittest
import numpy as np
from learning.topp import topp


class Test_Topp(unittest.TestCase):

    def testNormalBehaviour(self):
        v = np.array([0.95, 0.05])
        expected = np.exp(v) / np.sum(np.exp(v))
        actual = topp(v)

        np.testing.assert_array_equal(expected, actual)


