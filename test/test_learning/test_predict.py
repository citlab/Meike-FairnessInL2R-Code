import unittest
from learning import predict


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.pathToTestData = '../../../../master_thesis/DELTR-multinomial/sample/synthetic/unittest_example/unittest_synthetic_binomial_test.csv'
        self.pathToModelFile = '../../../../master_thesis/DELTR-multinomial/sample/synthetic/unittest_example/model.npy'
        self.resultDir = '../../../../master_thesis/DELTR-multinomial/sample/synthetic/unittest_example/'
        self.protectedColumn = 0

    def test_predict_method(self):
        predictor = predict.DELTR_Predictor(self.pathToTestData,
                                            self.pathToModelFile,
                                            self.resultDir,
                                            self.protectedColumn)

        predictor.predict()


if __name__ == '__main__':
    unittest.main()
