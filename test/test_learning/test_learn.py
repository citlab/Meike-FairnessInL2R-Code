import unittest
from learning import train


class TestLearnClass(unittest.TestCase):
    def setUp(self):
        self.pathToTrainingData = '../../../../master_thesis/DELTR-multinomial/sample/synthetic/unittest_example/unittest_synthetic_binomial_training.csv'
        self.pathToModelFile = '../../../../master_thesis/DELTR-multinomial/sample/synthetic/unittest_example/model'
        self.resultDir = '../../../../master_thesis/DELTR-multinomial/sample/synthetic/unittest_example/'
        self.gamma = 10
        self.numIter = 10
        self.learningRate = 0.001
        self.protCol = 0
        self.protAttr = 1
        self.initVar = 0.01
        self.lambdaa = 0.01

    def test_train_method(self):
        learning = train.DELTR_Trainer(self.pathToTrainingData,
                                self.pathToModelFile,
                                self.resultDir,
                                self.gamma,
                                self.numIter,
                                self.learningRate,
                                self.protCol,
                                self.protAttr,
                                self.initVar,
                                self.lambdaa)
        learning.train(False)


if __name__ == '__main__':
    unittest.main()
