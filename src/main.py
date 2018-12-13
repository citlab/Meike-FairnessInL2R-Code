'''
Created on Apr 2, 2018

@author: meike.zehlike
'''
import argparse

from data_preparation import *
from learning.train import DELTR_Trainer
from learning.predict import DELTR_Predictor
from evaluation.evaluate import DELTR_Evaluator

# TODO: bash Skripte anpassen, sodass Experimente mit Python laufen
# TODO: Experimente mit Python ausführen und Result-Pfade in Evaluate anpassen


def main():
    # parse command line options
    parser = argparse.ArgumentParser(prog='Disparate Exposure in Learning To Rank',
                                     epilog="=== === === end === === ===")

    parser.add_argument("--create",
                        nargs=1,
                        choices=['synthetic',
                                 'law-all',
                                 'law-gender',
                                 'law-asian',
                                 'law-black',
                                 'law-hispanic',
                                 'law-mexican',
                                 'law-puertorican',
                                 'trec',
                                 'engineering-withSemiPrivate',
                                 'engineering-withoutSemiPrivate'],
                        help="creates datasets from raw data and writes them to disk")
    parser.add_argument("--train",
                        nargs=5,
                        metavar=('TRAINING DATA', 'MODEL', 'DIRECTORY', 'GAMMA', 'COLORBLIND'),
                        help="runs training phase of DELTR with given GAMMA for given DATASET, stores \
                              trained model into file MODEL. All results are stored into DIRECTORY. \
                              if COLORBLIND is True, excludes protected column for training")
    parser.add_argument("--predict",
                        nargs=3,
                        metavar=('TEST DATA', 'MODEL', 'DIRECTORY'),
                        help="reads MODEL from disk, calculates predictions for TEST DATA and stores\
                              resulting rankings into DIRECTORY")
    parser.add_argument("--evaluate",
                        nargs=1,
                        metavar='DATASET',
                        choices=['synthetic',
                                 'law-all',
                                 'law-gender',
                                 'law-asian',
                                 'law-black',
                                 'law-hispanic',
                                 'law-mexican',
                                 'law-puertorican',
                                 'trec',
                                 'engineering-gender-withSemiPrivate',
                                 'engineering-highschool-withSemiPrivate',
                                 'engineering-gender-withoutSemiPrivate',
                                 'engineering-highschool-withoutSemiPrivate'],
                        help="evaluates performance and fairness metrics for DATASET predictions")

    args = parser.parse_args()

    ################### argparse create #########################
    if args.create == ['synthetic']:
        # TODO: link synthetic data here
        print('not yet implemented')
    elif args.create == ['law-all']:
        data = lawStudentDatasetPreparation.prepareAllInOneDataForFAIR()
        data.to_csv('../experiments/LawStudents/LSAT_AllInOne.csv', index=False, header=True)
    elif args.create == ['law-gender']:
        train, test = lawStudentDatasetPreparation.prepareGenderData()
        train.to_csv('../experiments/LawStudents/gender/LawStudents_Gender_train.txt', index=False, header=False)
        test.to_csv('../experiments/LawStudents/gender/LawStudents_Gender_test.txt', index=False, header=False)
    elif args.create == ['law-asian']:
        train, test = lawStudentDatasetPreparation.prepareRaceData('Asian', 'White')
        train.to_csv('../experiments/LawStudents/asian/LawStudents_Race_train.txt', index=False, header=False)
        test.to_csv('../experiments/LawStudents/asian/LawStudents_Race_test.txt', index=False, header=False)
    elif args.create == ['law-black']:
        train, test = lawStudentDatasetPreparation.prepareRaceData('Black', 'White')
        train.to_csv('../experiments/LawStudents/black/LawStudents_Race_train.txt', index=False, header=False)
        test.to_csv('../experiments/LawStudents/black/LawStudents_Race_test.txt', index=False, header=False)
    elif args.create == ['law-hispanic']:
        train, test = lawStudentDatasetPreparation.prepareRaceData('Hispanic', 'White')
        train.to_csv('../experiments/LawStudents/hispanic/LawStudents_Race_train.txt', index=False, header=False)
        test.to_csv('../experiments/LawStudents/hispanic/LawStudents_Race_test.txt', index=False, header=False)
    elif args.create == ['law-mexican']:
        train, test = lawStudentDatasetPreparation.prepareRaceData('Mexican', 'White')
        train.to_csv('../experiments/LawStudents/mexican/LawStudents_Race_train.txt', index=False, header=False)
        test.to_csv('../experiments/LawStudents/mexican/LawStudents_Race_test.txt', index=False, header=False)
    elif args.create == ['law-puertorican']:
        train, test = lawStudentDatasetPreparation.prepareRaceData('Puertorican', 'White')
        train.to_csv('../experiments/LawStudents/puertorican/LawStudents_Race_train.txt', index=False, header=False)
        test.to_csv('../experiments/LawStudents/puertorican/LawStudents_Race_test.txt', index=False, header=False)
    elif args.create == ['trec']:
        TRECDataToL2R.prepare()
    elif args.create == ['engineering-withSemiPrivate']:
        chileDataSetToL2R.prepareWithSemi()
    elif args.create == ['engineering-withoutSemiPrivate']:
        chileDataSetToL2R.prepareNoSemi()

    #################argparse train#############################
    elif args.train:
        pathToTrainingData = args.train[0]
        pathToModelFile = args.train[1]
        resultDir = args.train[2]
        gamma = float(args.train[3])
        colorblind = True if args.train[4] == 'True' else False

        numIter = 3000
        learningRate = 0.001
        initVar = 0.01
        lambdaa = 0.001

        protCol = 1
        protAttr = 1

        trainer = DELTR_Trainer(pathToTrainingData,
                                pathToModelFile,
                                resultDir,
                                gamma,
                                numIter,
                                learningRate,
                                protCol,
                                protAttr,
                                initVar,
                                lambdaa)
        trainer.train(colorblind)

    #################### argparse predict #################################
    elif args.predict:
        pathToTestData = args.predict[0]
        pathToModelFile = args.predict[1]
        resultDir = args.predict[2]
        protCol = 1

        predictor = DELTR_Predictor(pathToTestData,
                                    pathToModelFile,
                                    resultDir,
                                    protCol)

        predictor.predict()

    #################### argparse evaluate ################################
    elif args.evaluate == ['synthetic']:
        # TODO: link synthetic data here
        raise NotImplementedError
    elif args.evaluate == ['law-gender']:
        resultDir = '../experiments/LawStudents/gender/results/'
        resultDir = '../octave-src/sample/LawStudents/gender/results/'
        binSize = 200
        protAttr = 1
        evaluator = DELTR_Evaluator('law-gender',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['law-asian']:
        resultDir = '../experiments/LawStudents/race_asian/results/'
        resultDir = '../octave-src/sample/LawStudents/race_asian/results/'
        binSize = 100
        protAttr = 1
        evaluator = DELTR_Evaluator('law-asian',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['law-black']:
        resultDir = '../experiments/LawStudents/race_black/results/'
        resultDir = '../octave-src/sample/LawStudents/race_black/results/'
        binSize = 200
        protAttr = 1
        evaluator = DELTR_Evaluator('law-black',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['law-hispanic']:
        resultDir = '../experiments/LawStudents/race_hispanic/results/'
        resultDir = '../octave-src/sample/LawStudents/race_hispanic/results/'
        binSize = 100
        protAttr = 1
        evaluator = DELTR_Evaluator('law-hispanic',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['law-mexican']:
        resultDir = '../experiments/LawStudents/race_mexican/results/'
        resultDir = '../octave-src/sample/LawStudents/race_mexican/results/'
        binSize = 100
        protAttr = 1
        evaluator = DELTR_Evaluator('law-mexican',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['law-puertorican']:
        resultDir = '../experiments/LawStudents/race_puertorican/results/'
        resultDir = '../octave-src/sample/LawStudents/race_puertorican/results/'
        binSize = 100
        protAttr = 1
        evaluator = DELTR_Evaluator('law-puertorican',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['trec']:
        resultDir = '../experiments/TREC/results/'
        resultDir = '../octave-src/sample/TREC/results/'
        binSize = 10
        protAttr = 1
        evaluator = DELTR_Evaluator('trec',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['engineering-gender-withSemiPrivate']:
        resultDir = '../experiments/EngineeringStudents/results/'
        resultDir = '../octave-src/sample/ChileUni/Semi/gender/results/'
        binSize = 10
        protAttr = 1
        evaluator = DELTR_Evaluator('engineering-gender-withSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['engineering-highschool-withSemiPrivate']:
        resultDir = '../experiments/EngineeringStudents/results/'
        resultDir = '../octave-src/sample/ChileUni/Semi/highschool/results/'
        binSize = 10
        protAttr = 1
        evaluator = DELTR_Evaluator('engineering-highschool-withSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['engineering-gender-withoutSemiPrivate']:
        resultDir = '../experiments/EngineeringStudents/results/'
        resultDir = '../octave-src/sample/ChileUni/NoSemi/gender/results/'
        binSize = 10
        protAttr = 1
        evaluator = DELTR_Evaluator('engineering-gender-withoutSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    elif args.evaluate == ['engineering-highschool-withoutSemiPrivate']:
        resultDir = '../experiments/EngineeringStudents/results/'
        resultDir = '../octave-src/sample/ChileUni/NoSemi/highschool/results/'
        binSize = 10
        protAttr = 1
        evaluator = DELTR_Evaluator('engineering-highschool-withoutSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()
    else:
        parser.error("choose one command line option")


if __name__ == '__main__':
    main()
