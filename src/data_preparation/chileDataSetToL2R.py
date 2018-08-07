'''
Created on May 7, 2018

@author: mzehlike
'''

import pandas as pd
import data_preparation.chileDatasetPreparation as prep


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
##############################################################################################
# WITHOUT SEMI-PRIVATE SCHOOLS
##############################################################################################

# write dataset for gender
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withoutSemiPrivate(data)
train_fold1, test_fold1, \
train_fold2, test_fold2, \
train_fold3, test_fold3, \
train_fold4, test_fold4, \
train_fold5, test_fold5 = prep.prepareForL2R(data)

train_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/chileDataL2R_gender_nosemi_fold1_train.txt', index=False, header=False)
test_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/chileDataL2R_gender_nosemi_fold1_test.txt', index=False, header=False)

train_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/chileDataL2R_gender_nosemi_fold2_train.txt', index=False, header=False)
test_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/chileDataL2R_gender_nosemi_fold2_test.txt', index=False, header=False)

train_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/chileDataL2R_gender_nosemi_fold3_train.txt', index=False, header=False)
test_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/chileDataL2R_gender_nosemi_fold3_test.txt', index=False, header=False)

train_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/chileDataL2R_gender_nosemi_fold4_train.txt', index=False, header=False)
test_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/chileDataL2R_gender_nosemi_fold4_test.txt', index=False, header=False)

train_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/chileDataL2R_gender_nosemi_fold5_train.txt', index=False, header=False)
test_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/chileDataL2R_gender_nosemi_fold5_test.txt', index=False, header=False)

# write dataset for highschool type
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withoutSemiPrivate(data)
train_fold1, test_fold1, \
train_fold2, test_fold2, \
train_fold3, test_fold3, \
train_fold4, test_fold4, \
train_fold5, test_fold5 = prep.prepareForL2R(data, gender=False)

train_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/chileDataL2R_highschool_nosemi_fold1_train.txt', index=False, header=False)
test_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/chileDataL2R_highschool_nosemi_fold1_test.txt', index=False, header=False)

train_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/chileDataL2R_highschool_nosemi_fold2_train.txt', index=False, header=False)
test_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/chileDataL2R_highschool_nosemi_fold2_test.txt', index=False, header=False)

train_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/chileDataL2R_highschool_nosemi_fold3_train.txt', index=False, header=False)
test_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/chileDataL2R_highschool_nosemi_fold3_test.txt', index=False, header=False)

train_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/chileDataL2R_highschool_nosemi_fold4_train.txt', index=False, header=False)
test_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/chileDataL2R_highschool_nosemi_fold4_test.txt', index=False, header=False)

train_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/chileDataL2R_highschool_nosemi_fold5_train.txt', index=False, header=False)
test_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/chileDataL2R_highschool_nosemi_fold5_test.txt', index=False, header=False)


# write dataset for colorblind rankings
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation_withoutSemiPrivate(data)
train_fold1, test_fold1, \
train_fold2, test_fold2, \
train_fold3, test_fold3, \
train_fold4, test_fold4, \
train_fold5, test_fold5f = prep.prepareForL2R(data, colorblind=True)

train_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_1/chileDataL2R_colorblind_nosemi_fold1_train.txt', index=False, header=False)
test_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_1/chileDataL2R_colorblind_nosemi_fold1_test.txt', index=False, header=False)

train_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_2/chileDataL2R_highschool_nosemi_fold2_train.txt', index=False, header=False)
test_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_2/chileDataL2R_highschool_nosemi_fold2_test.txt', index=False, header=False)

train_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_3/chileDataL2R_highschool_nosemi_fold3_train.txt', index=False, header=False)
test_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_3/chileDataL2R_highschool_nosemi_fold3_test.txt', index=False, header=False)

train_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_4/chileDataL2R_highschool_nosemi_fold4_train.txt', index=False, header=False)
test_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_4/chileDataL2R_highschool_nosemi_fold4_test.txt', index=False, header=False)

train_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_5/chileDataL2R_highschool_nosemi_fold5_train.txt', index=False, header=False)
test_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_5/chileDataL2R_highschool_nosemi_fold5_test.txt', index=False, header=False)


# TODO: remove this, dead code
# data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_train.csv', sep=' ',
#                    names=['query_id', 'gender', '1', '2', '3', '4', '5', 'score'])
# rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
# for queryName, group in rankingsPerQuery:
#     rowNum = group.shape[0]
#     newScores = (rowNum - group.index) / rowNum
#     group['score'] = newScores
# data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_train.csv', index=False, header=False)
#
# data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_test.csv', sep=' ',
#                    names=['query_id', '2', '3', '4', '5', '6', '7', 'score'])
# rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
# for queryName, group in rankingsPerQuery:
#     rowNum = group.shape[0]
#     group['score'] = (rowNum - group.index) / rowNum
# data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/features_with_total_order-withGender_withZscore_test.csv', index=False, header=False)
#
# data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_train.csv', sep=' ',
#                    names=['query_id', '2', '3', '4', '5', '6', '7', 'score'])
# rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
# for queryName, group in rankingsPerQuery:
#     rowNum = group.shape[0]
#     group['score'] = (rowNum - group.index) / rowNum
# data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_train.csv', index=False, header=False)
#
# data = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_test.csv', sep=' ',
#                    names=['query_id', '2', '3', '4', '5', '6', '7', 'score'])
# rankingsPerQuery = data.groupby(data['query_id'], as_index=False, sort=False)
# for queryName, group in rankingsPerQuery:
#     rowNum = group.shape[0]
#     group['score'] = (rowNum - group.index) / rowNum
# data.to_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/features_with_total_order-withGender_withZscore_test.csv', index=False, header=False)
