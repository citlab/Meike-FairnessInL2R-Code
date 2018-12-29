'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import os
from fileinput import filename


class DELTR_Evaluator():
    '''
    evaluates model that was trained on given dataset and writes evaluation into textfile
    indicators are:
        exposure of protected group in training scores and predictions
        exposure of non-protected group in training scores and predictions
        difference of group exposure in training scores and predictions
        mean ranking position of protected group in training scores and predictions
        mean ranking position of non-protected group in training scores and predictions
        median ranking position of protected group in training scores and predictions
        median ranking position of non-protected group in training scores and predictions
        precision at top 1, 5, 10, 20, 100, 200
        kendall's tau
    :field __trainingDir:             directory in which training scores and predictions are stored
    :field __resultDir:               path where to store the result files
    :field __protectedAttribute:      int, defines what the protected attribute in the dataset was
    :field __dataset:                 string, specifies which dataset to evaluate
    :field __chunkSize:               int, defines how many items belong to one chunk in the bar plots
    :field __columnNames:             predictions are read into dataframe with defined column names
    :field __predictions:              np-array with predicted scores
    :field __original:                np-array with training scores
    :field __evaluationFilename       string, filename to store the evaluation results (without path)
    :field __plotFilename             string, filename to store the barplots (without path)
    '''

    def __init__(self, dataset, resultDir, binSize, protAttr):
        self.__trainingDir = '../octave-src/sample/'
        self.__resultDir = resultDir
        if not os.path.exists(resultDir):
            os.makedirs(resultDir)
        self.__protectedAttribute = protAttr
        self.__dataset = dataset
        self.__chunkSize = binSize
        self.__columnNames = ["query_id", "doc_id", "prediction", "prot_attr"]

    def evaluate(self):
        if self.__dataset == 'synthetic':
            raise NotImplementedError
            # GAMMA 0
            scoreDir = self.__trainingDir + 'synthetic/top_male_bottom_female/GAMMA=0/'
            gamma = '0'
            self.__original = pd.read_csv(scoreDir + 'sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                          sep=",",
                                          names=self.__columnNames)
            self.__predictions = pd.read_csv(scoreDir + 'sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                             sep=",",
                                             names=self.__columnNames)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_per_query()
            self.__evaluate(synthetic=True)

            # GAMMA 75
            scoreDir = self.__trainingDir + 'synthetic/top_male_bottom_female/GAMMA=75/'
            gamma = 'small'
            self.__original = pd.read_csv(scoreDir + 'sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                          sep=",",
                                          names=self.__columnNames)
            self.__predictions = pd.read_csv(scoreDir + 'sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                             sep=",",
                                             names=self.__columnNames)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_per_query()
            self.__evaluate(synthetic=True)

            # GAMMA 150
            gamma = 'large'
            scoreDir = self.__trainingDir + 'synthetic/top_male_bottom_female/GAMMA=150/'
            self.__original = pd.read_csv(scoreDir + 'sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                          sep=",",
                                          names=self.__columnNames)
            self.__predictions = pd.read_csv(scoreDir + 'sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                             sep=",",
                                             names=self.__columnNames)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_per_query()
            self.__evaluate(synthetic=True)

        ###########################################################################################
        ###############H############################################################################

        elif self.__dataset == 'engineering-gender-withoutSemiPrivate':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'engineering-highschool-withoutSemiPrivate':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/PREPROCESSED/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = '0'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################
        elif self.__dataset == 'engineering-gender-withSemiPrivate':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_2/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_3/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_4/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_5/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_2/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_3/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_4/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/gender/fold_5/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'engineering-highschool-withSemiPrivate':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/COLORBLIND/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/FA-IR/',
                             self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'trec':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'TREC/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_5/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_6/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'TREC/fold_1/COLORBLIND/',
                             self.__trainingDir + 'TREC/fold_2/COLORBLIND/',
                             self.__trainingDir + 'TREC/fold_3/COLORBLIND/',
                             self.__trainingDir + 'TREC/fold_4/COLORBLIND/',
                             self.__trainingDir + 'TREC/fold_5/COLORBLIND/',
                             self.__trainingDir + 'TREC/fold_6/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'TREC/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_5/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_6/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'TREC/fold_1/PREPROCESSED/',
                             self.__trainingDir + 'TREC/fold_2/PREPROCESSED/',
                             self.__trainingDir + 'TREC/fold_3/PREPROCESSED/',
                             self.__trainingDir + 'TREC/fold_4/PREPROCESSED/',
                             self.__trainingDir + 'TREC/fold_5/PREPROCESSED/',
                             self.__trainingDir + 'TREC/fold_6/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = '0'
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_2/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_3/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_4/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_5/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_6/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_5/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_6/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_5/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_6/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/FA-IR/',
                             self.__trainingDir + 'TREC/fold_2/FA-IR/',
                             self.__trainingDir + 'TREC/fold_3/FA-IR/',
                             self.__trainingDir + 'TREC/fold_4/FA-IR/',
                             self.__trainingDir + 'TREC/fold_5/FA-IR/',
                             self.__trainingDir + 'TREC/fold_6/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-gender':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-asian':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-black':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-hispanic':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-mexican':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-puertorican':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/COLORBLIND/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/PREPROCESSED/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=0/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=SMALL/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=LARGE/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "p=01_"
            pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/FA-IR/']

            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=02_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=03_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=04_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=05_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=06_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=07_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=08_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            #--------------------------------------------------------------------------------------

            pString = "p=09_"
            self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

    def __prepareData(self, pathsToScores, pathsForColorblind=None, pString=""):
        '''
        reads training scores and predictions from disc and arranges them NICELY into a dataframe
        '''
        trainingfiles = (dirname + 'trainingScores_ORIG.pred' for dirname in pathsToScores)
        predictionFiles = (dirname + pString + 'predictions_SORTED.pred' for dirname in pathsToScores)
        trainingScores = pd.concat((pd.read_csv(file,
                                                sep=",",
                                                names=self.__columnNames) \
                                    for file in trainingfiles))

        predictedScores = pd.concat((pd.read_csv(file,
                                                sep=",",
                                                names=self.__columnNames) \
                                    for file in predictionFiles))

        if pathsForColorblind is not None:
            # if we want to evaluate a colorblind training, we have to put the protectedAttribute
            colorblindTrainingFiles = (dirname + 'trainingScores_ORIG.pred' for dirname in pathsForColorblind)
            trainingScoresWithProtected = pd.concat((pd.read_csv(file, sep=",", names=self.__columnNames) \
                                         for file in colorblindTrainingFiles))

            trainingScores, \
            predictedScores = self.__add_prot_to_colorblind(trainingScoresWithProtected,
                                                            trainingScores,
                                                            predictedScores)
        return trainingScores, predictedScores

    def __evaluate(self, synthetic=False):
        pd.set_option('display.float_format', lambda x: '%.3f' % x)

        predictedGroups = self.__predictions.groupby(self.__predictions['query_id'], as_index=True, sort=False)
#         originalGroups = self.__original.groupby(self.__predictions['query_id'], as_index=True, sort=False)
        originalGroups = self.__original.groupby(self.__original['query_id'], as_index=True, sort=False)

        result = pd.DataFrame(np.nan,
                              index=range(0, len(predictedGroups)),
                              columns=['query_id',
                                       'exposure_prot_pred', 'exposure_nprot_pred', 'exp_diff_pred',
                                       'exposure_prot_orig', 'exposure_nprot_orig', 'exp_diff_orig',
                                       'precision_top1', 'precision_top5', 'precision_top10',
                                       'precision_top20', 'precision_top100', 'precision_top500',
                                       'prot_pos_mean_pred', 'nprot_pos_mean_pred', 'prot_pos_median_pred', 'nprot_pos_median_pred',
                                       'prot_pos_mean_orig', 'nprot_pos_mean_orig', 'prot_pos_median_orig', 'nprot_pos_median_orig',
                                       'kendall_tau', 'p_value'])

        i = 0
        for name, predGroup in predictedGroups:
            print(name)
            origGroup = originalGroups.get_group(name)

            predGroup = predGroup.reset_index()
            origGroup = origGroup.reset_index()

            result.loc[i]['query_id'] = name
            result.loc[i]['exposure_prot_pred'] = self.__calculate_group_exposure(predGroup, origGroup)[0]
            result.loc[i]['exposure_nprot_pred'] = self.__calculate_group_exposure(predGroup, origGroup)[1]
            result.loc[i]['exp_diff_pred'] = self.__calculate_group_exposure(predGroup, origGroup)[2]
            result.loc[i]['exposure_prot_orig'] = self.__calculate_group_exposure(predGroup, origGroup)[3]
            result.loc[i]['exposure_nprot_orig'] = self.__calculate_group_exposure(predGroup, origGroup)[4]
            result.loc[i]['exp_diff_orig'] = self.__calculate_group_exposure(predGroup, origGroup)[5]
            result.loc[i]['prot_pos_mean_pred'] = self.__avg_group_position(predGroup)[0]
            result.loc[i]['nprot_pos_mean_pred'] = self.__avg_group_position(predGroup)[1]
            result.loc[i]['prot_pos_median_pred'] = self.__avg_group_position(predGroup)[2]
            result.loc[i]['nprot_pos_median_pred'] = self.__avg_group_position(predGroup)[3]
            result.loc[i]['prot_pos_mean_orig'] = self.__avg_group_position(origGroup)[0]
            result.loc[i]['nprot_pos_mean_orig'] = self.__avg_group_position(origGroup)[1]
            result.loc[i]['prot_pos_median_orig'] = self.__avg_group_position(origGroup)[2]
            result.loc[i]['nprot_pos_median_orig'] = self.__avg_group_position(origGroup)[3]
            result.loc[i]['precision_top1'] = self.__precision_at_position(predGroup, origGroup, 1, 'doc_id')
            result.loc[i]['precision_top5'] = self.__precision_at_position(predGroup, origGroup, 5, 'doc_id')
            result.loc[i]['precision_top10'] = self.__precision_at_position(predGroup, origGroup, 10, 'doc_id')
            result.loc[i]['precision_top20'] = self.__precision_at_position(predGroup, origGroup, 20, 'doc_id')
            if (not synthetic) :
                result.loc[i]['precision_top100'] = self.__precision_at_position(predGroup, origGroup, 100, 'doc_id')
                result.loc[i]['precision_top500'] = self.__precision_at_position(predGroup, origGroup, 500, 'doc_id')
            result.loc[i]['kendall_tau'] = stats.kendalltau(origGroup['doc_id'], predGroup['doc_id'])[0]
            result.loc[i]['p_value'] = stats.kendalltau(origGroup['doc_id'], predGroup['doc_id'])[1]
            i += 1

        result = result.mean()

        with open(self.__evaluationFilename, "w") as text_file:
            print(result, file=text_file)
        return

    def __precision_at_position(self, prediction, original, pos, mergeCol):
        top_pred = prediction.head(n=pos)
        top_orig = original.head(n=pos)

        sec = pd.merge(top_pred, top_orig, how='inner', on=mergeCol)
        precision = sec.shape[0] / pos
        return precision

    def __avg_group_position(self, prediction):
        prot_rows_pred = prediction.loc[prediction['prot_attr'] == self.__protectedAttribute]
        nprot_rows_pred = prediction.loc[prediction['prot_attr'] != self.__protectedAttribute]

        prot_pos_mean = np.mean(prot_rows_pred.index.values)
        nprot_pos_mean = np.mean(nprot_rows_pred.index.values)

        prot_pos_median = np.median(prot_rows_pred.index.values)
        nprot_pos_median = np.median(nprot_rows_pred.index.values)

        return prot_pos_mean, nprot_pos_mean, prot_pos_median, nprot_pos_median

    def __calculate_group_exposure(self, prediction, original):

        # exposure in predictions
        prot_rows_pred = prediction.loc[prediction['prot_attr'] == self.__protectedAttribute]
        nprot_rows_pred = prediction.loc[prediction['prot_attr'] != self.__protectedAttribute]

        prot_exp_per_doc = 1 / np.log(prot_rows_pred.index.values + 2)
        avg_prot_exp_pred = sum(prot_exp_per_doc) / prot_exp_per_doc.shape[0]

        nprot_exp_per_doc = 1 / np.log(nprot_rows_pred.index.values + 2)
        avg_nprot_exp_pred = sum(nprot_exp_per_doc) / nprot_rows_pred.shape[0]

        # exposure in original
        prot_rows_orig = original.loc[original['prot_attr'] == self.__protectedAttribute]
        nprot_rows_orig = original.loc[original['prot_attr'] != self.__protectedAttribute]

        prot_exp_per_doc = 1 / np.log(prot_rows_orig.index.values + 2)
        avg_prot_exp_orig = sum(prot_exp_per_doc) / prot_rows_orig.shape[0]

        nprot_exp_per_doc = 1 / np.log(nprot_rows_orig.index.values + 2)
        avg_nprot_exp_orig = sum(nprot_exp_per_doc) / nprot_rows_orig.shape[0]

        return avg_prot_exp_pred, avg_nprot_exp_pred, \
               (avg_nprot_exp_pred - avg_prot_exp_pred), \
               avg_prot_exp_orig, avg_nprot_exp_orig, \
               (avg_nprot_exp_orig - avg_prot_exp_orig)

    def _protected_percentage_per_chunk_per_query(self, ranking, plot_filename):
        '''
        calculates percentage of protected (non-protected resp.) for each chunk of the ranking
        plots them into a figure
        '''
        rankingsPerQuery = ranking.groupby(ranking['query_id'], as_index=False, sort=False)
        for name, rank in rankingsPerQuery:

            filename = plot_filename[:-4] + "_" + str(name) + plot_filename[-4:]

            rowNum = rank.shape[0]
            chunkStartPositions = np.arange(0, rowNum, self.__chunkSize)

            result_protected = np.empty(len(chunkStartPositions))
            result_nonprotected = np.empty(len(chunkStartPositions))

    #         total_protected = rank['prot_attr'].value_counts()[1]
    #         total_nonprotected = rank['prot_attr'].value_counts()[0]

            for idx, start in enumerate(chunkStartPositions):
                if idx == (len(chunkStartPositions) - 1):
                    # last Chunk
                    end = rowNum
                else:
                    end = chunkStartPositions[idx + 1]
                chunk = rank.iloc[start:end]

                try:
                    chunk_protected = chunk['prot_attr'].value_counts()[1]
                except KeyError:
                    # no protected elements in this chunk
                    chunk_protected = 0

                try:
                    chunk_nonprotected = chunk['prot_attr'].value_counts()[0]
                except KeyError:
                    # no nonprotected elements in this chunk
                    chunk_nonprotected = 0

                result_protected[idx] = chunk_protected / rowNum
                result_nonprotected[idx] = chunk_nonprotected / rowNum

            self.__plot_protected_percentage_per_chunk(result_protected,
                                                       result_nonprotected,
                                                       chunkStartPositions,
                                                       filename,
                                                       self.__chunkSize / 2)

    def __protected_percentage_per_chunk_average_all_queries(self):
        '''
        calculates percentage of protected (non-protected resp.) for each chunk of the ranking
        plots them into a figure

        averages results over all queries
        '''
        rankingsPerQuery = self.__predictions.groupby(self.__predictions['query_id'], as_index=False, sort=False)
        shortest_query = math.inf

        data_matriks = pd.DataFrame()

        for name, rank in rankingsPerQuery:
            # find shortest query
            if (len(rank) < shortest_query):
                shortest_query = len(rank)

        for name, rank in rankingsPerQuery:
            temp = rank['prot_attr'].head(shortest_query)
            data_matriks[name] = temp.reset_index(drop=True)
            print(data_matriks.shape)

        chunkStartPositions = np.arange(0, shortest_query, self.__chunkSize)

        result_protected = np.empty(len(chunkStartPositions))
        result_nonprotected = np.empty(len(chunkStartPositions))

    #         total_protected = rank['prot_attr'].value_counts()[1]
    #         total_nonprotected = rank['prot_attr'].value_counts()[0]

        for idx, start in enumerate(chunkStartPositions):
            if idx == (len(chunkStartPositions) - 1):
                # last Chunk
                end = shortest_query
            else:
                end = chunkStartPositions[idx + 1]
            chunk = data_matriks.iloc[start:end]

            chunk_protected = 0
            for col in chunk:
                try:
                    chunk_protected += chunk[col].value_counts()[1]
                except KeyError:
                    # no protected elements in this chunk
                    chunk_protected += 0

            chunk_nonprotected = 0
            for col in chunk:
                try:
                    chunk_nonprotected += chunk[col].value_counts()[0]
                except KeyError:
                    # no nonprotected elements in this chunk
                    chunk_nonprotected = 0

            result_protected[idx] = chunk_protected / shortest_query
            result_nonprotected[idx] = chunk_nonprotected / shortest_query

        self.__plot_protected_percentage_per_chunk(result_protected,
                                                   result_nonprotected,
                                                   chunkStartPositions,
                                                   self.__plotFilename,
                                                   self.__chunkSize / 2)

    def __plot_protected_percentage_per_chunk(self, prot, nonprot, x_ticks, plot_filename, bar_width):
        mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
        # avoid type 3 (i.e. bitmap) fonts in figures
        mpl.rcParams['ps.useafm'] = True
        mpl.rcParams['pdf.use14corefonts'] = True
        mpl.rcParams['text.usetex'] = True

        _, ax = plt.subplots(figsize=(20, 5))
    #     plt.plot(prot, 'r-')
    #     plt.plot(nonprot, 'b')
        width = bar_width

        ax.bar(x_ticks, prot, color='orangered', width=width)
        ax.bar(x_ticks + width, nonprot, color='b', width=width)

        # plt.xticks(np.arange(tick_length))
        # ax.set_xticklabels(x_ticks)

        plt.xlabel ("position");
        plt.ylabel("proportion")
    #     plt.legend(['protected', 'non-protected'])
    #     plt.show()

        plt.savefig(plot_filename, bbox_inches='tight')

    def __add_prot_to_colorblind(self, trainingScoresWithProtected, colorblind_orig, colorblind_pred):
        orig_prot_attr = trainingScoresWithProtected['prot_attr']
        print(colorblind_orig.index)
        print(orig_prot_attr.index)
        colorblind_orig["prot_attr"] = orig_prot_attr

        for doc_id in colorblind_orig['doc_id']:
            prot_status_for_pred = colorblind_orig.loc[colorblind_orig['doc_id'] == doc_id]['prot_attr'].values
            colorblind_pred.at[colorblind_pred['doc_id'] == doc_id, 'prot_attr'] = prot_status_for_pred

        return colorblind_orig, colorblind_pred

