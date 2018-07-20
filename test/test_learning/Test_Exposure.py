import unittest
import numpy as np
from src.learning import exposure

class TestExposureMethod(unittest.TestCase):

    @classmethod
    def setUp(self):
        self._all_items = np.array([5, 6, 7, 8, 2, 9, 10, 4, 3])
        self._prot_group = np.array([5, 6, 8, 4, 3])
        self._nprot_group = np.array([7, 1, 2, 9, 10])

    def tearDown(self):
        pass

    def test_exposure_for_each_group(self):
        expected_prot = 0.02987333554210711
        actual_prot = exposure.exposure(self._prot_group, self._all_items)
        np.testing.assert_equal(expected_prot, actual_prot)

        expected_nprot = 0.25866567263568563
        actual_nprot = exposure.exposure(self._nprot_group, self._all_items)
        np.testing.assert_equal(expected_nprot, actual_nprot)




