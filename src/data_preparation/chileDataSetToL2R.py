'''
Created on May 7, 2018

@author: mzehlike
'''

import pandas as pd
import util.chileDatasetPreparation as prep

# #############################################################################################
# # WITH SEMI-PRIVATE SCHOOLS
# #############################################################################################
#
# # write dataset for gender
# data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
# data = prep.principalDataPreparation_withSemiPrivate(data)
# train, test = prep.prepareForL2R(data)
#
# train.to_csv('../../data/ChileUniversity/chileDataL2R_gender_semi_train.txt', index=False, header=False)
# test.to_csv('../../data/ChileUniversity/chileDataL2R_gender_semi_test.txt', index=False, header=False)
#
# # write dataset for highschool type
# data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
# data = prep.principalDataPreparation_withSemiPrivate(data)
# train, test = prep.prepareForL2R(data, gender=False)
#
# train.to_csv('../../data/ChileUniversity/chileDataL2R_highschool_semi_train.txt', index=False, header=False)
# test.to_csv('../../data/ChileUniversity/chileDataL2R_highschool_semi_test.txt', index=False, header=False)
#
# # write dataset for colorblind rankings
# data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
# data = prep.principalDataPreparation_withSemiPrivate(data)
# train, test = prep.prepareForL2R(data, colorblind=True)
#
# train.to_csv('../../data/ChileUniversity/chileDataL2R_colorblind_semi_train.txt', index=False, header=False)
# test.to_csv('../../data/ChileUniversity/chileDataL2R_colorblind_semi_test.txt', index=False, header=False)
#
# ##############################################################################################
# # WITHOUT SEMI-PRIVATE SCHOOLS
# ##############################################################################################
#
# # write dataset for gender
# data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
# data = prep.principalDataPreparation_withoutSemiPrivate(data)
# train, test = prep.prepareForL2R(data)
#
# train.to_csv('../../data/ChileUniversity/chileDataL2R_gender_nosemi_train.txt', index=False, header=False)
# test.to_csv('../../data/ChileUniversity/chileDataL2R_gender_nosemi_test.txt', index=False, header=False)
#
# # write dataset for highschool type
# data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
# data = prep.principalDataPreparation_withoutSemiPrivate(data)
# train, test = prep.prepareForL2R(data, gender=False)
#
# train.to_csv('../../data/ChileUniversity/chileDataL2R_highschool_nosemi_train.txt', index=False, header=False)
# test.to_csv('../../data/ChileUniversity/chileDataL2R_highschool_nosemi_test.txt', index=False, header=False)
#
# # write dataset for colorblind rankings
# data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
# data = prep.principalDataPreparation_withoutSemiPrivate(data)
# train, test = prep.prepareForL2R(data, colorblind=True)
#
# train.to_csv('../../data/ChileUniversity/chileDataL2R_colorblind_nosemi_train.txt', index=False, header=False)
# test.to_csv('../../data/ChileUniversity/chileDataL2R_colorblind_nosemi_test.txt', index=False, header=False)

# TODO: remove this, dead code
data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_train.csv', sep=' ',
                   names=['query_id', 'gender', '1', '2', '3', '4', '5', 'score'])
rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
for queryName, group in rankingsPerQuery:
    rowNum = group.shape[0]
    newScores = (rowNum - group.index) / rowNum
    group['score'] = newScores
data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_train.csv', index=False, header=False)

data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_test.csv', sep=' ',
                   names=['query_id', '2', '3', '4', '5', '6', '7', 'score'])
rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
for queryName, group in rankingsPerQuery:
    rowNum = group.shape[0]
    group['score'] = (rowNum - group.index) / rowNum
data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_test.csv', index=False, header=False)

data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_train.csv', sep=' ',
                   names=['query_id', '2', '3', '4', '5', '6', '7', 'score'])
rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
for queryName, group in rankingsPerQuery:
    rowNum = group.shape[0]
    group['score'] = (rowNum - group.index) / rowNum
data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_train.csv', index=False, header=False)

data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_test.csv', sep=' ',
                   names=['query_id', '2', '3', '4', '5', '6', '7', 'score'])
rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
for queryName, group in rankingsPerQuery:
    rowNum = group.shape[0]
    group['score'] = (rowNum - group.index) / rowNum
data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_test.csv', index=False, header=False)
