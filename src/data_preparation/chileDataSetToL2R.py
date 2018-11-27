'''
Created on May 7, 2018

@author: mzehlike
'''

import pandas as pd
import data_preparation.chileDatasetPreparation as prep


def prepareWithSemi():
    # write dataset for gender
    data = pd.read_excel('../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
    data = prep.principalDataPreparation_withSemiPrivate(data)
    train_fold1, test_fold1, \
    train_fold2, test_fold2, \
    train_fold3, test_fold3, \
    train_fold4, test_fold4, \
    train_fold5, test_fold5 = prep.prepareForL2R(data)

    pathToGender = '../experiments/EngineeringStudents/SemiPrivate/gender/'

    train_fold1.to_csv(pathToGender + 'fold_1/chileDataL2R_gender_semi_fold1_train.txt', index=False, header=False)
    test_fold1.to_csv(pathToGender + 'fold_1/chileDataL2R_gender_semi_fold1_test.txt', index=False, header=False)

    train_fold2.to_csv(pathToGender + 'fold_2/chileDataL2R_gender_semi_fold2_train.txt', index=False, header=False)
    test_fold2.to_csv(pathToGender + 'fold_2/chileDataL2R_gender_semi_fold2_test.txt', index=False, header=False)

    train_fold3.to_csv(pathToGender + 'fold_3/chileDataL2R_gender_semi_fold3_train.txt', index=False, header=False)
    test_fold3.to_csv(pathToGender + 'fold_3/chileDataL2R_gender_semi_fold3_test.txt', index=False, header=False)

    train_fold4.to_csv(pathToGender + 'fold_4/chileDataL2R_gender_semi_fold4_train.txt', index=False, header=False)
    test_fold4.to_csv(pathToGender + 'fold_4/chileDataL2R_gender_semi_fold4_test.txt', index=False, header=False)

    train_fold5.to_csv(pathToGender + 'fold_5/chileDataL2R_gender_semi_fold5_train.txt', index=False, header=False)
    test_fold5.to_csv(pathToGender + 'fold_5/chileDataL2R_gender_semi_fold5_test.txt', index=False, header=False)

    # write dataset for highschool type
    data = pd.read_excel('../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
    data = prep.principalDataPreparation_withSemiPrivate(data)
    train_fold1, test_fold1, \
    train_fold2, test_fold2, \
    train_fold3, test_fold3, \
    train_fold4, test_fold4, \
    train_fold5, test_fold5 = prep.prepareForL2R(data, gender=False)

    pathToHighschool = '../experiments/EngineeringStudents/SemiPrivate/highschool/'

    train_fold1.to_csv(pathToHighschool + 'fold_1/chileDataL2R_highschool_semi_fold1_train.txt', index=False, header=False)
    test_fold1.to_csv(pathToHighschool + 'fold_1/chileDataL2R_highschool_semi_fold1_test.txt', index=False, header=False)

    train_fold2.to_csv(pathToHighschool + 'fold_2/chileDataL2R_highschool_semi_fold2_train.txt', index=False, header=False)
    test_fold2.to_csv(pathToHighschool + 'fold_2/chileDataL2R_highschool_semi_fold2_test.txt', index=False, header=False)

    train_fold3.to_csv(pathToHighschool + 'fold_3/chileDataL2R_highschool_semi_fold3_train.txt', index=False, header=False)
    test_fold3.to_csv(pathToHighschool + 'fold_3/chileDataL2R_highschool_semi_fold3_test.txt', index=False, header=False)

    train_fold4.to_csv(pathToHighschool + 'fold_4/chileDataL2R_highschool_semi_fold4_train.txt', index=False, header=False)
    test_fold4.to_csv(pathToHighschool + 'fold_4/chileDataL2R_highschool_semi_fold4_test.txt', index=False, header=False)

    train_fold5.to_csv(pathToHighschool + 'fold_5/chileDataL2R_highschool_semi_fold5_train.txt', index=False, header=False)
    test_fold5.to_csv(pathToHighschool + 'fold_5/chileDataL2R_highschool_semi_fold5_test.txt', index=False, header=False)


def prepareNoSemi():
    # write dataset for gender
    data = pd.read_excel('../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
    data = prep.principalDataPreparation_withoutSemiPrivate(data)
    train_fold1, test_fold1, \
    train_fold2, test_fold2, \
    train_fold3, test_fold3, \
    train_fold4, test_fold4, \
    train_fold5, test_fold5 = prep.prepareForL2R(data)

    pathToGender = '../experiments/EngineeringStudents/NoSemiPrivate/gender/'

    train_fold1.to_csv(pathToGender + 'fold_1/chileDataL2R_gender_nosemi_fold1_train.txt', index=False, header=False)
    test_fold1.to_csv(pathToGender + 'fold_1/chileDataL2R_gender_nosemi_fold1_test.txt', index=False, header=False)

    train_fold2.to_csv(pathToGender + 'fold_2/chileDataL2R_gender_nosemi_fold2_train.txt', index=False, header=False)
    test_fold2.to_csv(pathToGender + 'fold_2/chileDataL2R_gender_nosemi_fold2_test.txt', index=False, header=False)

    train_fold3.to_csv(pathToGender + 'fold_3/chileDataL2R_gender_nosemi_fold3_train.txt', index=False, header=False)
    test_fold3.to_csv(pathToGender + 'fold_3/chileDataL2R_gender_nosemi_fold3_test.txt', index=False, header=False)

    train_fold4.to_csv(pathToGender + 'fold_4/chileDataL2R_gender_nosemi_fold4_train.txt', index=False, header=False)
    test_fold4.to_csv(pathToGender + 'fold_4/chileDataL2R_gender_nosemi_fold4_test.txt', index=False, header=False)

    train_fold5.to_csv(pathToGender + 'fold_5/chileDataL2R_gender_nosemi_fold5_train.txt', index=False, header=False)
    test_fold5.to_csv(pathToGender + 'fold_5/chileDataL2R_gender_nosemi_fold5_test.txt', index=False, header=False)

    # write dataset for highschool type
    data = pd.read_excel('../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
    data = prep.principalDataPreparation_withoutSemiPrivate(data)
    train_fold1, test_fold1, \
    train_fold2, test_fold2, \
    train_fold3, test_fold3, \
    train_fold4, test_fold4, \
    train_fold5, test_fold5 = prep.prepareForL2R(data, gender=False)

    pathToHighschool = '../experiments/EngineeringStudents/NoSemiPrivate/highschool/'

    train_fold1.to_csv(pathToHighschool + 'fold_1/chileDataL2R_highschool_nosemi_fold1_train.txt', index=False, header=False)
    test_fold1.to_csv(pathToHighschool + 'fold_1/chileDataL2R_highschool_nosemi_fold1_test.txt', index=False, header=False)

    train_fold2.to_csv(pathToHighschool + 'fold_2/chileDataL2R_highschool_nosemi_fold2_train.txt', index=False, header=False)
    test_fold2.to_csv(pathToHighschool + 'fold_2/chileDataL2R_highschool_nosemi_fold2_test.txt', index=False, header=False)

    train_fold3.to_csv(pathToHighschool + 'fold_3/chileDataL2R_highschool_nosemi_fold3_train.txt', index=False, header=False)
    test_fold3.to_csv(pathToHighschool + 'fold_3/chileDataL2R_highschool_nosemi_fold3_test.txt', index=False, header=False)

    train_fold4.to_csv(pathToHighschool + 'fold_4/chileDataL2R_highschool_nosemi_fold4_train.txt', index=False, header=False)
    test_fold4.to_csv(pathToHighschool + 'fold_4/chileDataL2R_highschool_nosemi_fold4_test.txt', index=False, header=False)

    train_fold5.to_csv(pathToHighschool + 'fold_5/chileDataL2R_highschool_nosemi_fold5_train.txt', index=False, header=False)
    test_fold5.to_csv(pathToHighschool + 'fold_5/chileDataL2R_highschool_nosemi_fold5_test.txt', index=False, header=False)

#     # write dataset for colorblind rankings
#     data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
#     data = prep.principalDataPreparation_withoutSemiPrivate(data)
#     train_fold1, test_fold1, \
#     train_fold2, test_fold2, \
#     train_fold3, test_fold3, \
#     train_fold4, test_fold4, \
#     train_fold5, test_fold5 = prep.prepareForL2R(data, colorblind=True)
#
#     train_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_1/chileDataL2R_colorblind_nosemi_fold1_train.txt', index=False, header=False)
#     test_fold1.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_1/chileDataL2R_colorblind_nosemi_fold1_test.txt', index=False, header=False)
#
#     train_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_2/chileDataL2R_highschool_nosemi_fold2_train.txt', index=False, header=False)
#     test_fold2.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_2/chileDataL2R_highschool_nosemi_fold2_test.txt', index=False, header=False)
#
#     train_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_3/chileDataL2R_highschool_nosemi_fold3_train.txt', index=False, header=False)
#     test_fold3.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_3/chileDataL2R_highschool_nosemi_fold3_test.txt', index=False, header=False)
#
#     train_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_4/chileDataL2R_highschool_nosemi_fold4_train.txt', index=False, header=False)
#     test_fold4.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_4/chileDataL2R_highschool_nosemi_fold4_test.txt', index=False, header=False)
#
#     train_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_5/chileDataL2R_highschool_nosemi_fold5_train.txt', index=False, header=False)
#     test_fold5.to_csv('../../octave-src/sample/ChileUni/NoSemi/colorblind/fold_5/chileDataL2R_highschool_nosemi_fold5_test.txt', index=False, header=False)

