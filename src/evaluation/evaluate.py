'''
Created on May 11, 2018

@author: mzehlike
'''

from adjustText import adjust_text
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import math
import os
import re
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
    :field __predictions:             np-array with predicted scores
    :field __original:                np-array with training scores
    :field __experimentNamesAndFiles: collects experiment names and result filenames to use for scatter plot
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
        self.__experimentNamesAndFiles = {}

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

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename

            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
            self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=0/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=SMALL/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=LARGE/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename

            #######################################################################################
            # FA*IR as post-processing evaluation
            pString = "_p_"
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/FA-IR/']

            self.__original, self.__predictions, p_share = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-post-p*"] = self.__evaluationFilename

            #--------------------------------------------------------------------------------------

            pString = "p-"

            self.__original, self.__predictions, p_minus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p-"] = self.__evaluationFilename
            #--------------------------------------------------------------------------------------

            pString = "p+"
            self.__original, self.__predictions, p_plus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p+"] = self.__evaluationFilename

            ######################################################################################

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

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-pre-p*"] = self.__evaluationFilename
            #######################################################################################
            gamma = 'PREPROCESSED_PMinus'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/PREPROCESSED_PMinus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-pre-p-"] = self.__evaluationFilename
            #######################################################################################
            gamma = 'PREPROCESSED_PPlus'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/gender/fold_1/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_2/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_3/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_4/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/gender/fold_5/PREPROCESSED_PPlus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-pre-p+"] = self.__evaluationFilename

            #######################################################################################
            utility1, utilityLabel1 = "kendall-tau", "Kendall's Tau"
            utility2, utilityLabel2 = "precision-top100", "Precision Top 100"
            fairness1P, fairnessLabel1 = "exposure-prot-pred", "Group Exposure"
            fairness1NP = "exposure-nprot-pred"
            fairness2P, fairnessLabel2 = "prot-pos-median-pred", "Group Median Position"
            fairness2NP = "nprot-pos-median-pred"

            legendLabelDict = {'colorblind' : {'label': 'Colorblind L2R', 'marker': '', 'color': 'red'},
                               'gamma=0' : {'label': 'Standard L2R', 'marker': '', 'color': 'blue'},
                               'gamma=small' : {'label': 'DELTR Small Gamma', 'marker': '$D^{-}$', 'color': '#00b300'},
                               'gamma=large' : {'label': 'DELTR Large Gamma', 'marker': '$D^{+}$', 'color': '#006600'},
                               'fair-post-p*' : {'label': str('FA*IR post-processing $p^{*}$'), 'marker': '$F^{*}$', 'color': '#4da9ff'},
                               'fair-post-p+' : {'label': str('FA*IR post-processing $p^{+}$'), 'marker': '$F^{+}$', 'color': '#0069cc'},
                               'fair-post-p-' : {'label': str('FA*IR post-processing $p^{-}$'), 'marker': '$F^{-}$', 'color': '#99ceff'},
                                'fair-pre-p*': {'label': str('FA*IR pre-processing $p^{*}$'), 'marker': '$\overline{F}^{*}$', 'color': '#F97B06'},
                                'fair-pre-p+': {'label': str('FA*IR pre-processing $p^{+}$'), 'marker': '$\overline{F}^{+}$', 'color': '#C25D00'},
                                'fair-pre-p-': {'label': str('FA*IR pre-processing $p^{-}$'), 'marker': '$\overline{F}^{-}$', 'color': '#FF9A3D'}}

            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness1P, fairness1NP, utilityLabel1, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness2P, fairness2NP, utilityLabel1, fairnessLabel2, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness1P, fairness1NP, utilityLabel2, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness2P, fairness2NP, utilityLabel2, fairnessLabel2, legendLabelDict)

            # self.__experimentNamesAndFiles.pop('fair-post-p*')
            # self.__experimentNamesAndFiles.pop('fair-post-p-')
            # self.__experimentNamesAndFiles.pop('fair-post-p+')
            #
            #
            # legendLabelDict = {'colorblind': 'Colorblind L2R',
            #                    'gamma=0': 'Standard L2R',
            #                    'gamma=small': 'DELTR Small Gamma',
            #                    'gamma=large': 'DELTR Large Gamma',
            #                    'fair-pre-p*': str('FA*IR $p^{*}=' + p_share + '$'),
            #                    'fair-pre-p+': str('FA*IR $p^{+}=' + p_plus + '$'),
            #                    'fair-pre-p-': str('FA*IR $p^{-}=' + p_minus + '$')}
            #
            # scatterFilename = self.__resultDir + 'scatter_PREPROCESSED_' + utility1 + '-' + fairness1P + self.__dataset + '.png'
            # self.__scatterPlot(scatterFilename, utility1, fairness1P, fairness1NP, utilityLabel1, fairnessLabel1,
            #                    legendLabelDict)
            # scatterFilename = self.__resultDir + 'scatter_PREPROCESSED_' + utility1 + '-' + fairness2P + self.__dataset + '.png'
            # self.__scatterPlot(scatterFilename, utility1, fairness2P, fairness2NP, utilityLabel1, fairnessLabel2,
            #                    legendLabelDict)
            # scatterFilename = self.__resultDir + 'scatter_PREPROCESSED_' + utility2 + '-' + fairness1P + self.__dataset + '.png'
            # self.__scatterPlot(scatterFilename, utility2, fairness1P, fairness1NP, utilityLabel2, fairnessLabel1,
            #                    legendLabelDict)
            # scatterFilename = self.__resultDir + 'scatter_PREPROCESSED_' + utility2 + '-' + fairness2P + self.__dataset + '.png'
            # self.__scatterPlot(scatterFilename, utility2, fairness2P, fairness2NP, utilityLabel2, fairnessLabel2,
            #                legendLabelDict)

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

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)

            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename

            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
            self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)

            #######################################################################################
            gamma = '0'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=0/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=0/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=SMALL/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=LARGE/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "_p_"
            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/FA-IR/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/FA-IR/']

            self.__original, self.__predictions, p_share = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-post-p*"] = self.__evaluationFilename

            #--------------------------------------------------------------------------------------

            pString = "p-"

            self.__original, self.__predictions, p_minus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p-"] = self.__evaluationFilename
            #--------------------------------------------------------------------------------------

            pString = "p+"
            self.__original, self.__predictions, p_plus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p+"] = self.__evaluationFilename

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

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p*"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED_PMinus'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/PREPROCESSED_PMinus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p-"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED_PPlus'
            pathsForColorblind = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_1/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_2/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_3/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_4/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'ChileUni/NoSemi/highschool/fold_5/PREPROCESSED_PPlus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p+"] = self.__evaluationFilename

            #######################################################################################

            utility1, utilityLabel1 = "kendall-tau", "Kendall's Tau"
            utility2, utilityLabel2 = "precision-top100", "Precision Top 100"
            fairness1P, fairnessLabel1 = "exposure-prot-pred", "Group Exposure"
            fairness1NP = "exposure-nprot-pred"
            fairness2P, fairnessLabel2 = "prot-pos-median-pred", "Group Median Position"
            fairness2NP = "nprot-pos-median-pred"

            legendLabelDict = {'colorblind' : {'label': 'Colorblind L2R', 'marker': '', 'color': 'red'},
                               'gamma=0' : {'label': 'Standard L2R', 'marker': '', 'color': 'blue'},
                               'gamma=small' : {'label': 'DELTR Small Gamma', 'marker': '$D^{-}$', 'color': '#00b300'},
                               'gamma=large' : {'label': 'DELTR Large Gamma', 'marker': '$D^{+}$', 'color': '#006600'},
                               'fair-post-p*' : {'label': str('FA*IR post-processing $p^{*}$'), 'marker': '$F^{*}$', 'color': '#4da9ff'},
                               'fair-post-p+' : {'label': str('FA*IR post-processing $p^{+}$'), 'marker': '$F^{+}$', 'color': '#0069cc'},
                               'fair-post-p-' : {'label': str('FA*IR post-processing $p^{-}$'), 'marker': '$F^{-}$', 'color': '#99ceff'},
                                'fair-pre-p*': {'label': str('FA*IR pre-processing $p^{*}$'), 'marker': '$\overline{F}^{*}$', 'color': '#F97B06'},
                                'fair-pre-p+': {'label': str('FA*IR pre-processing $p^{+}$'), 'marker': '$\overline{F}^{+}$', 'color': '#C25D00'},
                                'fair-pre-p-': {'label': str('FA*IR pre-processing $p^{-}$'), 'marker': '$\overline{F}^{-}$', 'color': '#FF9A3D'}}

            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness1P, fairness1NP, utilityLabel1, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness2P, fairness2NP, utilityLabel1, fairnessLabel2, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness1P, fairness1NP, utilityLabel2, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness2P, fairness2NP, utilityLabel2, fairnessLabel2, legendLabelDict)

        #         #########################################################################################
#         ##########################################################################################
#          elif self.__dataset == 'engineering-gender-withSemiPrivate':
#             gamma = 'colorblind'
#             pathsForColorblind = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=0/']
#
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_2/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_3/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_4/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_5/COLORBLIND/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
#
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename
#
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
#             self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)
#
#             #######################################################################################
#
#             gamma = '0'
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=0/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'small'
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=SMALL/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'large'
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_2/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_3/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_4/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_5/GAMMA=LARGE/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename
#
#             #######################################################################################
#             # FA*IR as post-processing evaluation
#
#             pString = "p=p*"
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/gender/fold_1/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_2/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_3/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_4/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/gender/fold_5/FA-IR/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["fair-post-p=p*"] = self.__evaluationFilename
#
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*-01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*-01"] = self.__evaluationFilename
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*+01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*+01"] = self.__evaluationFilename
#
#             #######################################################################################
#             utility1, utility2, fairness1, fairness2 = "kendall-tau", "precision-top100", "exposure-prot-pred", "prot-pos-median-pred"
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness2)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness2)
#
#         ###########################################################################################
#         ###########################################################################################
#
#         elif self.__dataset == 'engineering-highschool-withSemiPrivate':
#             gamma = 'colorblind'
#             pathsForColorblind = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=0/',
#                                   self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=0/']
#
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/COLORBLIND/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/COLORBLIND/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename
#
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
#             self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)
#
#             #######################################################################################
#
#             gamma = '0'
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=0/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=0/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'small'
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=SMALL/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=SMALL/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'large'
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/GAMMA=LARGE/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/GAMMA=LARGE/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename
#
#             #######################################################################################
#             # FA*IR as post-processing evaluation
#
#             pString = "p=p*"
#             pathsToScores = [self.__trainingDir + 'ChileUni/Semi/highschool/fold_1/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_2/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_3/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_4/FA-IR/',
#                              self.__trainingDir + 'ChileUni/Semi/highschool/fold_5/FA-IR/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["fair-post-p=p*"] = self.__evaluationFilename
#
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*-01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*-01"] = self.__evaluationFilename
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*+01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*+01"] = self.__evaluationFilename
#
#             #######################################################################################
#             utility1, utility2, fairness1, fairness2 = "kendall-tau", "precision-top100", "exposure-prot-pred", "prot-pos-median-pred"
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness2)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness2)

        ##########################################################################################
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

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename

            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
            self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_2/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_3/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_4/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_5/GAMMA=0/',
                             self.__trainingDir + 'TREC/fold_6/GAMMA=0/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_2/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_3/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_4/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_5/GAMMA=SMALL/',
                             self.__trainingDir + 'TREC/fold_6/GAMMA=SMALL/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_2/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_3/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_4/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_5/GAMMA=LARGE/',
                             self.__trainingDir + 'TREC/fold_6/GAMMA=LARGE/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "_p_"
            pathsToScores = [self.__trainingDir + 'TREC/fold_1/FA-IR/',
                             self.__trainingDir + 'TREC/fold_2/FA-IR/',
                             self.__trainingDir + 'TREC/fold_3/FA-IR/',
                             self.__trainingDir + 'TREC/fold_4/FA-IR/',
                             self.__trainingDir + 'TREC/fold_5/FA-IR/',
                             self.__trainingDir + 'TREC/fold_6/FA-IR/']

            self.__original, self.__predictions, p_share = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-post-p*"] = self.__evaluationFilename

            #--------------------------------------------------------------------------------------

            pString = "p-"

            self.__original, self.__predictions, p_minus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p-"] = self.__evaluationFilename
            #--------------------------------------------------------------------------------------

            pString = "p+"
            self.__original, self.__predictions, p_plus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p+"] = self.__evaluationFilename

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

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p*"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED_PMinus'
            pathsForColorblind = [self.__trainingDir + 'TREC/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_5/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_6/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'TREC/fold_1/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'TREC/fold_2/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'TREC/fold_3/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'TREC/fold_4/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'TREC/fold_5/PREPROCESSED_PMinus/',
                             self.__trainingDir + 'TREC/fold_6/PREPROCESSED_PMinus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p-"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED_PPlus'
            pathsForColorblind = [self.__trainingDir + 'TREC/fold_1/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_2/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_3/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_4/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_5/GAMMA=0/',
                                  self.__trainingDir + 'TREC/fold_6/GAMMA=0/']

            pathsToScores = [self.__trainingDir + 'TREC/fold_1/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'TREC/fold_2/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'TREC/fold_3/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'TREC/fold_4/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'TREC/fold_5/PREPROCESSED_PPlus/',
                             self.__trainingDir + 'TREC/fold_6/PREPROCESSED_PPlus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p+"] = self.__evaluationFilename
            #######################################################################################

            utility1, utilityLabel1 = "precision-top20", "Precision Top 20"
            utility2, utilityLabel2 = "precision-top10", "Precision Top 10"
            fairness1P, fairnessLabel1 = "exposure-prot-pred", "Group Exposure"
            fairness1NP = "exposure-nprot-pred"
            fairness2P, fairnessLabel2 = "prot-pos-median-pred", "Group Median Position"
            fairness2NP = "nprot-pos-median-pred"

            legendLabelDict = {'colorblind' : {'label': 'Colorblind L2R', 'marker': '', 'color': 'red'},
                               'gamma=0' : {'label': 'Standard L2R', 'marker': '', 'color': 'blue'},
                               'gamma=small' : {'label': 'DELTR Small Gamma', 'marker': '$D^{-}$', 'color': '#00b300'},
                               'gamma=large' : {'label': 'DELTR Large Gamma', 'marker': '$D^{+}$', 'color': '#006600'},
                               'fair-post-p*' : {'label': str('FA*IR post-processing $p^{*}$'), 'marker': '$F^{*}$', 'color': '#4da9ff'},
                               'fair-post-p+' : {'label': str('FA*IR post-processing $p^{+}$'), 'marker': '$F^{+}$', 'color': '#0069cc'},
                               'fair-post-p-' : {'label': str('FA*IR post-processing $p^{-}$'), 'marker': '$F^{-}$', 'color': '#99ceff'},
                                'fair-pre-p*': {'label': str('FA*IR pre-processing $p^{*}$'), 'marker': '$\overline{F}^{*}$', 'color': '#F97B06'},
                                'fair-pre-p+': {'label': str('FA*IR pre-processing $p^{+}$'), 'marker': '$\overline{F}^{+}$', 'color': '#C25D00'},
                                'fair-pre-p-': {'label': str('FA*IR pre-processing $p^{-}$'), 'marker': '$\overline{F}^{-}$', 'color': '#FF9A3D'}}

            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness1P, fairness1NP, utilityLabel1, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness2P, fairness2NP, utilityLabel1, fairnessLabel2, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness1P, fairness1NP, utilityLabel2, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness2P, fairness2NP, utilityLabel2, fairnessLabel2, legendLabelDict)

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-gender':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/COLORBLIND/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename

            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
            self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/GAMMA=SMALL/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/GAMMA=LARGE/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "_p_"
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/FA-IR/']

            self.__original, self.__predictions, p_share = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-post-p*"] = self.__evaluationFilename

            #--------------------------------------------------------------------------------------

            pString = "p-"

            self.__original, self.__predictions, p_minus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p-"] = self.__evaluationFilename
            #--------------------------------------------------------------------------------------

            pString = "p+"
            self.__original, self.__predictions, p_plus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p+"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/PREPROCESSED/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p*"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED_PMinus'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/PREPROCESSED_PMinus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p-"] = self.__evaluationFilename

            ######################################################################################
            gamma = 'PREPROCESSED_PPlus'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/gender/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/gender/PREPROCESSED_PPlus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p+"] = self.__evaluationFilename
            #######################################################################################

            utility1, utilityLabel1 = "kendall-tau", "Kendall's Tau"
            utility2, utilityLabel2 = "precision-top500", "Precision Top 500"
            fairness1P, fairnessLabel1 = "exposure-prot-pred", "Group Exposure"
            fairness1NP = "exposure-nprot-pred"
            fairness2P, fairnessLabel2 = "prot-pos-median-pred", "Group Median Position"
            fairness2NP = "nprot-pos-median-pred"

            legendLabelDict = {'colorblind' : {'label': 'Colorblind L2R', 'marker': '', 'color': 'red'},
                               'gamma=0' : {'label': 'Standard L2R', 'marker': '', 'color': 'blue'},
                               'gamma=small' : {'label': 'DELTR Small Gamma', 'marker': '$D^{-}$', 'color': '#00b300'},
                               'gamma=large' : {'label': 'DELTR Large Gamma', 'marker': '$D^{+}$', 'color': '#006600'},
                               'fair-post-p*' : {'label': str('FA*IR post-processing $p^{*}$'), 'marker': '$F^{*}$', 'color': '#4da9ff'},
                               'fair-post-p+' : {'label': str('FA*IR post-processing $p^{+}$'), 'marker': '$F^{+}$', 'color': '#0069cc'},
                               'fair-post-p-' : {'label': str('FA*IR post-processing $p^{-}$'), 'marker': '$F^{-}$', 'color': '#99ceff'},
                               'fair-pre-p*': {'label': str('FA*IR pre-processing $p^{*}$'), 'marker': '$\overline{F}^{*}$', 'color': '#F97B06'},
                               'fair-pre-p+': {'label': str('FA*IR pre-processing $p^{+}$'), 'marker': '$\overline{F}^{+}$', 'color': '#C25D00'},
                               'fair-pre-p-': {'label': str('FA*IR pre-processing $p^{-}$'), 'marker': '$\overline{F}^{-}$', 'color': '#FF9A3D'}}

            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness1P, fairness1NP, utilityLabel1, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness2P, fairness2NP, utilityLabel1, fairnessLabel2, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness1P, fairness1NP, utilityLabel2, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness2P, fairness2NP, utilityLabel2, fairnessLabel2, legendLabelDict)

#         ###########################################################################################
#
#         elif self.__dataset == 'law-asian':
#             gamma = 'colorblind'
#             pathsForColorblind = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=0/']
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/COLORBLIND/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename
#
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
#             self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)
#
#             #######################################################################################
#
#             gamma = '0'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=0/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'small'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=SMALL/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'large'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/GAMMA=LARGE/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename
#
#             #######################################################################################
#             # FA*IR as post-processing evaluation
#
#             pString = "p=p*"
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_asian/FA-IR/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["fair-post-p=p*"] = self.__evaluationFilename
#
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*-01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*-01"] = self.__evaluationFilename
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*+01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*+01"] = self.__evaluationFilename
#
#             #######################################################################################
#             utility1, utility2, fairness1, fairness2 = "kendall-tau", "precision-top500", "exposure-prot-pred", "prot-pos-median-pred"
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness2)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness2)

        ###########################################################################################
        ###########################################################################################

        elif self.__dataset == 'law-black':
            gamma = 'colorblind'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/COLORBLIND/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename

            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
            self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)

            #######################################################################################

            gamma = '0'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'small'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/GAMMA=SMALL/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename

            #######################################################################################

            gamma = 'large'
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/GAMMA=LARGE/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename

            #######################################################################################
            # FA*IR as post-processing evaluation

            pString = "_p_"
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/FA-IR/']

            self.__original, self.__predictions, p_share = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()

            self.__experimentNamesAndFiles["fair-post-p*"] = self.__evaluationFilename

            #--------------------------------------------------------------------------------------
            # we didn't do p- because p* was 0.06 and p- would have been too low
            #--------------------------------------------------------------------------------------

            pString = "p+"
            self.__original, self.__predictions, p_plus = self.__prepareData(pathsToScores, p=pString)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-post-p+"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/PREPROCESSED/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p*"] = self.__evaluationFilename

            #######################################################################################
            gamma = 'PREPROCESSED_PPlus'
            pathsForColorblind = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']
            pathsToScores = [self.__trainingDir + 'LawStudents/race_black/PREPROCESSED_PPlus/']

            self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'

            self.__protected_percentage_per_chunk_average_all_queries()
            self.__evaluate()
            self.__experimentNamesAndFiles["fair-pre-p+"] = self.__evaluationFilename

            #######################################################################################

            utility1, utilityLabel1 = "kendall-tau", "Kendall's Tau"
            utility2, utilityLabel2 = "precision-top500", "Precision Top 500"
            fairness1P, fairnessLabel1 = "exposure-prot-pred", "Group Exposure"
            fairness1NP = "exposure-nprot-pred"
            fairness2P, fairnessLabel2 = "prot-pos-median-pred", "Group Median Position"
            fairness2NP = "nprot-pos-median-pred"

            legendLabelDict = {'colorblind' : {'label': 'Colorblind L2R', 'marker': '', 'color': 'red'},
                               'gamma=0' : {'label': 'Standard L2R', 'marker': '', 'color': 'blue'},
                               'gamma=small' : {'label': 'DELTR Small Gamma', 'marker': '$D^{-}$', 'color': '#00b300'},
                               'gamma=large' : {'label': 'DELTR Large Gamma', 'marker': '$D^{+}$', 'color': '#006600'},
                               'fair-post-p*' : {'label': str('FA*IR post-processing $p^{*}$'), 'marker': '$F^{*}$', 'color': '#4da9ff'},
                               'fair-post-p+' : {'label': str('FA*IR post-processing $p^{+}$'), 'marker': '$F^{+}$', 'color': '#0069cc'},
                                'fair-pre-p*': {'label': str('FA*IR pre-processing $p^{*}$'), 'marker': '$\overline{F}^{*}$', 'color': '#F97B06'},
                                'fair-pre-p+': {'label': str('FA*IR pre-processing $p^{+}$'), 'marker': '$\overline{F}^{+}$', 'color': '#C25D00'}}

            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness1P, fairness1NP, utilityLabel1, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility1 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility1, fairness2P, fairness2NP, utilityLabel1, fairnessLabel2, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness1P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness1P, fairness1NP, utilityLabel2, fairnessLabel1, legendLabelDict)
            scatterFilename = self.__resultDir + 'scatter_' + utility2 + '-' + fairness2P + self.__dataset + '.png'
            self.__scatterPlot(scatterFilename, utility2, fairness2P, fairness2NP, utilityLabel2, fairnessLabel2, legendLabelDict)

            # #######################################################################################
            # gamma = 'PREPROCESSED_PMinus'
            # pathsForColorblind = [self.__trainingDir + 'LawStudents/race_black/GAMMA=0/']
            # pathsToScores = [self.__trainingDir + 'LawStudents/race_black/PREPROCESSED_PMinus/']
            #
            # self.__original, self.__predictions, _ = self.__prepareData(pathsToScores, pathsForColorblind)
            # self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
            # self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
            #
            # self.__protected_percentage_per_chunk_average_all_queries()
            # self.__evaluate()
#         ###########################################################################################
#         ###########################################################################################
#
#         elif self.__dataset == 'law-hispanic':
#             gamma = 'colorblind'
#             pathsForColorblind = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=0/']
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/COLORBLIND/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename
#
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
#             self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)
#
#             #######################################################################################
#
#             gamma = '0'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=0/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'small'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=SMALL/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'large'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/GAMMA=LARGE/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename
#
#             #######################################################################################
#             # FA*IR as post-processing evaluation
#
#             pString = "p=p*"
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_hispanic/FA-IR/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["fair-post-p=p*"] = self.__evaluationFilename
#
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*-01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*-01"] = self.__evaluationFilename
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*+01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*+01"] = self.__evaluationFilename
#
#             #######################################################################################
#             utility1, utility2, fairness1, fairness2 = "kendall-tau", "precision-top500", "exposure-prot-pred", "prot-pos-median-pred"
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness2)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness2)
#
#         ###########################################################################################
#         ###########################################################################################
#
#         elif self.__dataset == 'law-mexican':
#             gamma = 'colorblind'
#             pathsForColorblind = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=0/']
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/COLORBLIND/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename
#
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
#             self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)
#
#             #######################################################################################
#
#             gamma = '0'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=0/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'small'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=SMALL/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'large'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/GAMMA=LARGE/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename
#
#             #######################################################################################
#             # FA*IR as post-processing evaluation
#
#             pString = "p=p*"
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_mexican/FA-IR/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["fair-post-p=p*"] = self.__evaluationFilename
#
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*-01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*-01"] = self.__evaluationFilename
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*+01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*+01"] = self.__evaluationFilename
#
#             #######################################################################################
#             utility1, utility2, fairness1, fairness2 = "kendall-tau", "precision-top500", "exposure-prot-pred", "prot-pos-median-pred"
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness2)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness2)
#
#         ###########################################################################################
#         ###########################################################################################
#
#         elif self.__dataset == 'law-puertorican':
#             gamma = 'colorblind'
#             pathsForColorblind = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=0/']
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/COLORBLIND/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["colorblind"] = self.__evaluationFilename
#
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_groundtruth' + '_' + self.__dataset + '.png'
#             self.__protected_percentage_per_chunk_average_all_queries(plotGroundTruth=True)
#
#             #######################################################################################
#
#             gamma = '0'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=0/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=0"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'small'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=SMALL/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=small"] = self.__evaluationFilename
#
#             #######################################################################################
#
#             gamma = 'large'
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/GAMMA=LARGE/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_Gamma=' + gamma + '_' + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_Gamma=' + gamma + '_' + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["gamma=large"] = self.__evaluationFilename
#
#             #######################################################################################
#             # FA*IR as post-processing evaluation
#
#             pString = "p=p*"
#             pathsToScores = [self.__trainingDir + 'LawStudents/race_puertorican/FA-IR/']
#
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#
#             self.__experimentNamesAndFiles["fair-post-p=p*"] = self.__evaluationFilename
#
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*-01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*-01"] = self.__evaluationFilename
#             #--------------------------------------------------------------------------------------
#
#             pString = "p=p*+01"
#             self.__original, self.__predictions = self.__prepareData(pathsToScores, pString=pString)
#             self.__evaluationFilename = self.__resultDir + 'performanceResults_FAIR_' + pString + self.__dataset + '.txt'
#             self.__plotFilename = self.__resultDir + 'protNonprotDistribution_FAIR_' + pString + self.__dataset + '.png'
#
#             self.__protected_percentage_per_chunk_average_all_queries()
#             self.__evaluate()
#             self.__experimentNamesAndFiles["fair-post-p=p*+01"] = self.__evaluationFilename
#
#             #######################################################################################
#             utility1, utility2, fairness1, fairness2 = "kendall-tau", "precision-top500", "exposure-prot-pred", "prot-pos-median-pred"
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility1 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility1, fairness2)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness1 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness1)
#             scatterFilename = self.__resultDir + 'scatter_' + '_' + utility2 + '-' + fairness2 + self.__dataset + '.png'
#             self.__scatterPlot(scatterFilename, utility2, fairness2)

    def __prepareData(self, pathsToScores, pathsForColorblind=None, p=None):
        '''
        reads training scores and predictions from disc and arranges them NICELY into a dataframe
        '''
        collectedPs = []
        one_p = ""
        trainingfiles = list()
        predictionfiles = list()
        for dirName in pathsToScores:
            for _, _, filenames in os.walk(dirName):
                for fileName in filenames:
                    if 'trainingScores_ORIG.pred' in fileName:
                        trainingfiles.append(str(dirName + fileName))
                    if p is not None:
                        if 'predictions_SORTED.pred' and p in fileName:
                            one_p = re.findall(r'\d+.\d+', fileName)[0]
                            collectedPs.append(float(one_p))
                            predictionfiles.append(str(dirName + fileName))
                    else:
                        if 'predictions_SORTED.pred' in fileName:
                            predictionfiles.append(str(dirName + fileName))

        trainingScores = pd.concat((pd.read_csv(file,
                                                sep=",",
                                                names=self.__columnNames) \
                                    for file in trainingfiles))
        predictedScores = pd.concat((pd.read_csv(file,
                                                sep=",",
                                                names=self.__columnNames) \
                                    for file in predictionfiles))

        if pathsForColorblind is not None:
            # if we want to evaluate a colorblind training, we have to put the protectedAttribute
            colorblindTrainingFiles = (dirname + 'trainingScores_ORIG.pred' for dirname in pathsForColorblind)
            trainingScoresWithProtected = pd.concat((pd.read_csv(file, sep=",", names=self.__columnNames) \
                                         for file in colorblindTrainingFiles))

            trainingScores, \
            predictedScores = self.__add_prot_to_colorblind(trainingScoresWithProtected,
                                                            trainingScores,
                                                            predictedScores)

        return trainingScores, predictedScores, str(round(np.mean(collectedPs), 3))

    def __evaluate(self, synthetic=False):
        pd.set_option('display.float_format', lambda x: '%.3f' % x)

        predictedGroups = self.__predictions.groupby(self.__predictions['query_id'], as_index=True, sort=False)
#         originalGroups = self.__original.groupby(self.__predictions['query_id'], as_index=True, sort=False)
        originalGroups = self.__original.groupby(self.__original['query_id'], as_index=True, sort=False)

        result = pd.DataFrame(np.nan,
                              index=range(0, len(predictedGroups)),
                              columns=['query-id',
                                       'exposure-prot-pred', 'exposure-nprot-pred', 'exp-diff-pred',
                                       'exposure-prot-orig', 'exposure-nprot-orig', 'exp-diff-orig',
                                       'precision-top1', 'precision-top5', 'precision-top10',
                                       'precision-top20', 'precision-top100', 'precision-top500',
                                       'prot-pos-mean-pred', 'nprot-pos-mean-pred', 'prot-pos-median-pred', 'nprot-pos-median-pred',
                                       'prot-pos-mean-orig', 'nprot-pos-mean-orig', 'prot-pos-median-orig', 'nprot-pos-median-orig',
                                       'kendall-tau', 'p-value'])

        i = 0
        for name, predGroup in predictedGroups:
            origGroup = originalGroups.get_group(name)

            predGroup = predGroup.reset_index()
            origGroup = origGroup.reset_index()

            result.loc[i]['query-id'] = name
            result.loc[i]['exposure-prot-pred'] = self.__calculate_group_exposure(predGroup, origGroup)[0]
            result.loc[i]['exposure-nprot-pred'] = self.__calculate_group_exposure(predGroup, origGroup)[1]
            result.loc[i]['exp-diff-pred'] = self.__calculate_group_exposure(predGroup, origGroup)[2]
            result.loc[i]['exposure-prot-orig'] = self.__calculate_group_exposure(predGroup, origGroup)[3]
            result.loc[i]['exposure-nprot-orig'] = self.__calculate_group_exposure(predGroup, origGroup)[4]
            result.loc[i]['exp-diff-orig'] = self.__calculate_group_exposure(predGroup, origGroup)[5]
            result.loc[i]['prot-pos-mean-pred'] = self.__avg_group_position(predGroup)[0]
            result.loc[i]['nprot-pos-mean-pred'] = self.__avg_group_position(predGroup)[1]
            result.loc[i]['prot-pos-median-pred'] = self.__avg_group_position(predGroup)[2]
            result.loc[i]['nprot-pos-median-pred'] = self.__avg_group_position(predGroup)[3]
            result.loc[i]['prot-pos-mean-orig'] = self.__avg_group_position(origGroup)[0]
            result.loc[i]['nprot-pos-mean-orig'] = self.__avg_group_position(origGroup)[1]
            result.loc[i]['prot-pos-median-orig'] = self.__avg_group_position(origGroup)[2]
            result.loc[i]['nprot-pos-median-orig'] = self.__avg_group_position(origGroup)[3]
            result.loc[i]['precision-top1'] = self.__precision_at_position(predGroup, origGroup, 1, 'doc_id')
            result.loc[i]['precision-top5'] = self.__precision_at_position(predGroup, origGroup, 5, 'doc_id')
            result.loc[i]['precision-top10'] = self.__precision_at_position(predGroup, origGroup, 10, 'doc_id')
            result.loc[i]['precision-top20'] = self.__precision_at_position(predGroup, origGroup, 20, 'doc_id')
            if (not synthetic) :
                result.loc[i]['precision-top100'] = self.__precision_at_position(predGroup, origGroup, 100, 'doc_id')
                result.loc[i]['precision-top500'] = self.__precision_at_position(predGroup, origGroup, 500, 'doc_id')
            result.loc[i]['kendall-tau'] = stats.kendalltau(origGroup['doc_id'], predGroup['doc_id'])[0]
            result.loc[i]['p-value'] = stats.kendalltau(origGroup['doc_id'], predGroup['doc_id'])[1]
            i += 1

        result = result.mean()

        with open(self.__evaluationFilename, "w") as text_file:
            # write human readable
            print(result, file=text_file)

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

    def __protected_percentage_per_chunk_average_all_queries(self, plotGroundTruth=False):
        '''
        calculates percentage of protected (non-protected resp.) for each chunk of the ranking
        plots them into a figure

        averages results over all queries
        '''
        if plotGroundTruth:
            rankingsPerQuery = self.__original.groupby(self.__original['query_id'], as_index=False, sort=False)
        else:
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

        _, ax = plt.subplots(figsize=(17, 3))
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
        colorblind_orig["prot_attr"] = orig_prot_attr

        for doc_id in colorblind_orig['doc_id']:
            prot_status_for_pred = colorblind_orig.loc[colorblind_orig['doc_id'] == doc_id]['prot_attr'].values
            colorblind_pred.at[colorblind_pred['doc_id'] == doc_id, 'prot_attr'] = prot_status_for_pred

        return colorblind_orig, colorblind_pred

    def __scatterPlot(self, filename, utilityMeasure, fairnessMeasureProtected, fairnessMeasureNonProtected, utilLabel, fairLabel, legendLabelDict):
        print(utilLabel)
        print(fairLabel)
        createPlotFrame = True
        for key, value in self.__experimentNamesAndFiles.items():
            print(key)
            print(value)
            data = pd.read_table(value, delim_whitespace=True, header=None)
            # drop last row
            data = data[:-1]
            if createPlotFrame:
                columnNames = data[0].tolist()
                columnNames.insert(0, "experimentName")
                plotFrame = pd.DataFrame(columns=columnNames)
                createPlotFrame = False
            rowToAppend = data[1].tolist()
            rowToAppend.insert(0, key)
            plotFrame.loc[len(plotFrame)] = rowToAppend

        mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
        # avoid type 3 (i.e. bitmap) fonts in figures
        mpl.rcParams['ps.useafm'] = True
        mpl.rcParams['pdf.use14corefonts'] = True
        mpl.rcParams['text.usetex'] = True

        tick_spacing = 0.02

        _, ax = plt.subplots()
        xCol = plotFrame[utilityMeasure].apply(pd.to_numeric)
        yCol = plotFrame[fairnessMeasureProtected].apply(pd.to_numeric).div(plotFrame[fairnessMeasureNonProtected].apply(pd.to_numeric))

        deltr_small = None
        deltr_big = None

        fair_post_minus = None
        fair_post_share = None
        fair_post_plus = None

        fair_pre_minus = None
        fair_pre_share = None
        fair_pre_plus = None

        plot_text = []  # [0 for x in range(len(plotFrame['experimentName']))]
        # plot all protected/nonprotected
        for i, l in enumerate(plotFrame['experimentName']):
            x = xCol[i]
            y = yCol[i]
            m = legendLabelDict.get(l)['marker']
            if m:
                readableLabel = m + ': ' + legendLabelDict.get(l)['label']
            else:
                readableLabel = legendLabelDict.get(l)['label']
            color = legendLabelDict.get(l)['color']

            plt.plot(y, x, zorder=1)
            plot_text.append(plt.text(x, y, m, color=color, fontsize=20))

            if l == "colorblind":
                ax.scatter(x, y, label=readableLabel, s=100, linewidth=1, c=color, marker="s", zorder=2)
            elif l == "gamma=0":
                ax.scatter(x, y, label=readableLabel, s=100, linewidth=1, c=color, marker="X", zorder=2)
            else:
                ax.scatter(x, y, label=readableLabel, s=100, linewidth=1, marker='.', zorder=2, facecolors=color, edgecolors=color)
            # ax.annotate(m, (x, y), xytext=(x + 0.001, y + 0.001))

            if l == "gamma=small":
                deltr_small = i
            if l == "gamma=large":
                deltr_big = i

            if l == "fair-post-p-":
                fair_post_minus = i
            if l == "fair-post-p*":
                fair_post_share = i
            if l == "fair-post-p+":
                fair_post_plus = i

            if l == "fair-pre-p-":
                fair_pre_minus = i
            if l == "fair-pre-p*":
                fair_pre_share = i
            if l == "fair-pre-p+":
                fair_pre_plus = i

        # plot deltr lines
        ax.plot([xCol[deltr_small], xCol[deltr_big]], [yCol[deltr_small], yCol[deltr_big]], markersize=5., color="#00b300",
                    zorder=1, linewidth=1.5)

        if "law-black" in filename:
            # plot fair post lines
            ax.plot([xCol[fair_post_share], xCol[fair_post_plus]],
                    [yCol[fair_post_share], yCol[fair_post_plus]], markersize=5.,
                    color="#4da9ff", zorder=1, linewidth=1.5)

            # plot fair pre lines
            ax.plot([xCol[fair_pre_share], xCol[fair_pre_plus]],
                    [yCol[fair_pre_share], yCol[fair_pre_plus]],
                    markersize=5., color="#F97B06", zorder=1, linewidth=1.5)
        else:
            # plot fair post lines
            ax.plot([xCol[fair_post_minus], xCol[fair_post_share], xCol[fair_post_plus]],
                    [yCol[fair_post_minus], yCol[fair_post_share], yCol[fair_post_plus]], markersize=5., color="#4da9ff", zorder=1, linewidth=1.5)

            # plot fair pre lines
            ax.plot([xCol[fair_pre_minus], xCol[fair_pre_share], xCol[9]], [yCol[fair_pre_minus], yCol[fair_pre_share], yCol[fair_pre_plus]],
                    markersize=5., color="#F97B06", zorder=1, linewidth=1.5)

        factor = 1.0
        if "gender-withoutSemiPrivate" in filename:
            factor = 1.0

        if "highschool-withoutSemiPrivate" in filename:
            factor = 2.0

        if "law-gender" in filename:
            factor = 1.0

        if "law-black" in filename:
            factor = 1.5

        std_x = np.array(xCol).std(axis=0)
        mean_x = np.array(xCol).mean(axis=0)
        print(std_x)
        ax.set_xlim(left=(mean_x - 2 * std_x), right=(mean_x + 2 * std_x))

        ax.set_xlim(left=(mean_x - factor * std_x), right=(mean_x + factor * std_x))

        std_y = np.array(yCol).std(axis=0)
        mean_y = np.array(yCol).mean(axis=0)
        print(std_y)
        ax.set_ylim(bottom=(mean_y - 2 * std_y), top=(mean_y + 2 * std_y))
        ax.set_ylim(bottom=(mean_y - factor * std_y), top=(mean_y + factor * std_y))

        if "trec" in filename:
            factor = 0.7
            ax.set_xlim(left=(mean_x - factor * std_x), right=(mean_x + factor * std_x))
            ax.set_ylim(bottom=(mean_y - factor * std_y + 0.01), top=(mean_y + factor * std_y + 0.01))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.grid()
        plt.xlabel(utilLabel)
        plt.ylabel(fairLabel)

        adjust_text(plot_text)

        plt.savefig(filename, bbox_inches='tight', dpi=300)

