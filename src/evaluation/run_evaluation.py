'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd
from evaluation import evaluate

SYNTHETIC = 0

CHILE_GENDER_SEMI = 0
CHILE_HIGHSCHOOL_SEMI = 0
CHILE_GENDER_NOSEMI = 0
CHILE_HIGHSCHOOL_NOSEMI = 0

TREC = 0
TREC_BIG = 0

LAW_STUDENTS_GENDER = 1
LAW_STUDENTS_ASIAN = 0
LAW_STUDENTS_BLACK = 0
LAW_STUDENTS_HISPANIC = 0
LAW_STUDENTS_MEXICAN = 0
LAW_STUDENTS_PUERTORICAN = 0


def add_prot_to_colorblind(orig_scores, colorblind_orig, colorblind_pred):

    orig_prot_attr = orig_scores['prot_attr']
    colorblind_orig["prot_attr"] = orig_prot_attr

    for doc_id in colorblind_orig['doc_id']:
        prot_status_for_pred = colorblind_orig.loc[colorblind_orig['doc_id'] == doc_id]['prot_attr'].values
        colorblind_pred.at[colorblind_pred['doc_id'] == doc_id, 'prot_attr'] = prot_status_for_pred

    return colorblind_orig, colorblind_pred


#############################################################################################
# SYNTHETIC
#############################################################################################

if SYNTHETIC:
    chunksize = 5

    # GAMMA 0

    synthetic_male_top_0_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
    synthetic_male_top_0_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(synthetic_male_top_0_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/protected_percentage_per_chunk_per_query.png');
    evaluate.evaluate(synthetic_male_top_0_pred,
                      synthetic_male_top_0_orig,
                      '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/kendalls_tau_per_query.txt',
                      synthetic=True)

    # GAMMA 75

    synthetic_male_top_75_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
    synthetic_male_top_75_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(synthetic_male_top_75_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/protected_percentage_per_chunk_per_query.png');
    evaluate.evaluate(synthetic_male_top_75_pred,
                      synthetic_male_top_75_orig,
                      '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/kendalls_tau_per_query.txt',
                      synthetic=True)

    # GAMMA 150

    synthetic_male_top_150_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
    synthetic_male_top_150_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(synthetic_male_top_150_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/protected_percentage_per_chunk_per_query.png');
    evaluate.evaluate(synthetic_male_top_150_pred,
                      synthetic_male_top_150_orig,
                      '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/kendalls_tau_per_query.txt',
                      synthetic=True)


###############################################################################################
# CHILE UNIVERSITY GENDER WITH SEMI-PRIVATE
###############################################################################################

# if CHILE_GENDER_SEMI:
#     chunksize = 30
#
#     # COLORBLIND TRAINING
#     orig_scores = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_gender_test.txt_ORIG.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     chile_colorblind_gender_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_ORIG.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_colorblind_gender_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_SORTED.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     chile_colorblind_gender_orig, chile_colorblind_gender_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_gender_orig, chile_colorblind_gender_pred)
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_colorblind_gender_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/UNI-Colorblind-protected_percentage_per_chunk_gender_semi.png');
#     evaluate.evaluate(chile_colorblind_gender_pred, chile_colorblind_gender_orig, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/kendalls_tau_per_query_gender_semi.txt')
#
#     # GAMMA 0
#     chile_gender_0_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_gender_test.txt_ORIG.pred',
#                                     sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_gender_0_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_gender_test.txt_SORTED.pred',
#                                     sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_gender_0_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/UNI-Gamma_0-protected_percentage_per_chunk_gender_semi.png');
#     evaluate.evaluate(chile_gender_0_pred, chile_gender_0_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/kendalls_tau_per_query_gender_semi.txt')
#
#
#     # GAMMA 100000
#     chile_gender_gammaSmall_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_gender_test.txt_ORIG.pred',
#                                     sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_gender_gammaSmall_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_gender_test.txt_SORTED.pred',
#                                     sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_gender_gammaSmall_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/UNI-Gamma_100000-protected_percentage_per_chunk_gender_semi.png');
#     evaluate.evaluate(chile_gender_gammaSmall_pred, chile_gender_gammaSmall_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/kendalls_tau_per_query_gender_semi.txt')
#
#     # GAMMA 5000000
#     chile_gender_5000000_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_gender_test.txt_ORIG.pred',
#                                     sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_gender_5000000_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_gender_test.txt_SORTED.pred',
#                                     sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_gender_5000000_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/UNI-Gamma_5000000-protected_percentage_per_chunk_gender_semi.png');
#     evaluate.evaluate(chile_gender_5000000_pred, chile_gender_5000000_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/kendalls_tau_per_query_gender_semi.txt')

###############################################################################################
# CHILE UNIVERSITY HIGHSCHOOL WITH SEMI-PRIVATE
###############################################################################################

# if CHILE_HIGHSCHOOL_SEMI:
#     chunksize = 30
#
#     # COLORBLIND TRAINING
#     orig_scores = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_highschool_test.txt_ORIG.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     chile_colorblind_highschool_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_ORIG.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_colorblind_highschool_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_SORTED.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     chile_colorblind_highschool_orig, chile_colorblind_highschool_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_highschool_orig, chile_colorblind_highschool_pred)
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_colorblind_highschool_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/UNI-Colorblind-protected_percentage_per_chunk_highschool_semi.png');
#     evaluate.evaluate(chile_colorblind_highschool_pred, chile_colorblind_highschool_orig, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/kendalls_tau_per_query_highschool_semi.txt')
#
#     # GAMMA 0
#     chile_highschool_0_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_highschool_test.txt_ORIG.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_highschool_0_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_highschool_test.txt_SORTED.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_highschool_0_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/UNI-Gamma_0-protected_percentage_per_chunk_highschool_semi.png');
#     evaluate.evaluate(chile_highschool_0_pred, chile_highschool_0_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/kendalls_tau_per_query_highschool_semi.txt')
#
#
#     # GAMMA 100000
#     chile_highschool_100000_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_highschool_test.txt_ORIG.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_highschool_100000_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_highschool_test.txt_SORTED.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_highschool_100000_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/UNI-Gamma_100000-protected_percentage_per_chunk_highschool_semi.png');
#     evaluate.evaluate(chile_highschool_100000_pred, chile_highschool_100000_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/kendalls_tau_per_query_highschool_semi.txt')
#
#     # GAMMA 5000000
#     chile_highschool_5000000_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_highschool_test.txt_ORIG.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#     chile_highschool_5000000_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_highschool_test.txt_SORTED.pred',
#                                       sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
#     evaluate.protected_percentage_per_chunk_per_query(chile_highschool_5000000_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/UNI-Gamma_5000000-protected_percentage_per_chunk_highschool_semi.png');
#     evaluate.evaluate(chile_highschool_5000000_pred, chile_highschool_5000000_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/kendalls_tau_per_query_highschool_semi.txt')



###############################################################################################
# CHILE UNIVERSITY GENDER WITHOUT SEMI-PRIVATE
###############################################################################################

if CHILE_GENDER_NOSEMI:
    chunksize = 20

    # COLORBLIND TRAINING
    orig_files_with_gender = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/GAMMA=0/trainingScores_ORIG.pred']

    colorblind_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/COLORBLIND/trainingScores_ORIG.pred']

    colorblind_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/COLORBLIND/predictions_SORTED.pred']

    orig_scores = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                             for file in orig_files_with_gender))

    chile_colorblind_gender_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in colorblind_orig_files))


    chile_colorblind_gender_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in colorblind_pred_files))

    chile_colorblind_gender_orig, chile_colorblind_gender_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_gender_orig, chile_colorblind_gender_pred)

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_colorblind_gender_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/gender/results/UNI-Colorblind-protected_percentage_per_chunk_gender_nosemi.png');
    evaluate.evaluate(chile_colorblind_gender_pred, chile_colorblind_gender_orig, '../../octave-src/sample/ChileUni/NoSemi/gender/results/evaluation_gender_nosemi_colorblind.txt')

    # GAMMA 0

    gamma0_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/GAMMA=0/trainingScores_ORIG.pred']

    gamma0_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/GAMMA=0/predictions_SORTED.pred']


    chile_gender_0_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gamma0_orig_files))
    chile_gender_0_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gamma0_pred_files))

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_gender_0_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/gender/results/UNI-Gamma_0-protected_percentage_per_chunk_gender_nosemi.png');
    evaluate.evaluate(chile_gender_0_pred, chile_gender_0_orig, '../../octave-src/sample/ChileUni/NoSemi/gender/results/evaluation_gender_nosemi_gamma0.txt')


    # GAMMA SMALL

    gammaSmall_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/GAMMA=SMALL/trainingScores_ORIG.pred']

    gammaSmall_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/GAMMA=SMALL/predictions_SORTED.pred']


    chile_gender_gammaSmall_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaSmall_orig_files))
    chile_gender_gammaSmall_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaSmall_pred_files))

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_gender_gammaSmall_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/gender/results/UNI-Gamma_small-protected_percentage_per_chunk_gender_nosemi.png');
    evaluate.evaluate(chile_gender_gammaSmall_pred, chile_gender_gammaSmall_orig, '../../octave-src/sample/ChileUni/NoSemi/gender/results/evaluation_gender_nosemi_gammaSmall.txt')

    # GAMMA LARGE

    gammaLarge_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/GAMMA=LARGE/trainingScores_ORIG.pred']

    gammaLarge_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/gender/fold_1/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_2/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_3/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_4/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/gender/fold_5/GAMMA=LARGE/predictions_SORTED.pred']

    chile_gender_gammaLarge_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaLarge_orig_files))
    chile_gender_gammaLarge_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaLarge_pred_files))

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_gender_gammaLarge_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/gender/results/UNI-Gamma_large-protected_percentage_per_chunk_gender_nosemi.png');
    evaluate.evaluate(chile_gender_gammaLarge_pred, chile_gender_gammaLarge_orig, '../../octave-src/sample/ChileUni/NoSemi/gender/results/evaluation_gender_nosemi_gammaLarge.txt')

###############################################################################################
# CHILE UNIVERSITY HIGHSCHOOL WITHOUT SEMI-PRIVATE
###############################################################################################

if CHILE_HIGHSCHOOL_NOSEMI:
    chunksize = 20

        # COLORBLIND TRAINING
    orig_files_with_highschool = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/GAMMA=0/trainingScores_ORIG.pred',
                              '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/GAMMA=0/trainingScores_ORIG.pred']

    colorblind_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/COLORBLIND/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/COLORBLIND/trainingScores_ORIG.pred']

    colorblind_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/COLORBLIND/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/COLORBLIND/predictions_SORTED.pred']

    orig_scores = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                             for file in orig_files_with_highschool))

    chile_colorblind_highschool_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in colorblind_orig_files))


    chile_colorblind_highschool_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in colorblind_pred_files))

    chile_colorblind_highschool_orig, chile_colorblind_highschool_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_highschool_orig, chile_colorblind_highschool_pred)

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_colorblind_highschool_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/UNI-Colorblind-protected_percentage_per_chunk_highschool_nosemi.png');
    evaluate.evaluate(chile_colorblind_highschool_pred, chile_colorblind_highschool_orig, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/evaluation_highschool_nosemi_colorblind.txt')

    # GAMMA 0

    gamma0_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/GAMMA=0/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/GAMMA=0/trainingScores_ORIG.pred']

    gamma0_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/GAMMA=0/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/GAMMA=0/predictions_SORTED.pred']


    chile_highschool_0_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gamma0_orig_files))
    chile_highschool_0_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gamma0_pred_files))

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_highschool_0_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/UNI-Gamma_0-protected_percentage_per_chunk_highschool_nosemi.png');
    evaluate.evaluate(chile_highschool_0_pred, chile_highschool_0_orig, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/evaluation_highschool_nosemi_gamma0.txt')


    # GAMMA SMALL

    gammaSmall_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/GAMMA=SMALL/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/GAMMA=SMALL/trainingScores_ORIG.pred']

    gammaSmall_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/GAMMA=SMALL/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/GAMMA=SMALL/predictions_SORTED.pred']


    chile_highschool_gammaSmall_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaSmall_orig_files))
    chile_highschool_gammaSmall_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaSmall_pred_files))

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_highschool_gammaSmall_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/UNI-Gamma_small-protected_percentage_per_chunk_highschool_nosemi.png');
    evaluate.evaluate(chile_highschool_gammaSmall_pred, chile_highschool_gammaSmall_orig, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/evaluation_highschool_nosemi_gammaSmall.txt')

    # GAMMA LARGE

    gammaLarge_orig_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/GAMMA=LARGE/trainingScores_ORIG.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/GAMMA=LARGE/trainingScores_ORIG.pred']

    gammaLarge_pred_files = ['../../octave-src/sample/ChileUni/NoSemi/highschool/fold_1/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_2/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_3/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_4/GAMMA=LARGE/predictions_SORTED.pred',
                             '../../octave-src/sample/ChileUni/NoSemi/highschool/fold_5/GAMMA=LARGE/predictions_SORTED.pred']

    chile_highschool_gammaLarge_orig = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaLarge_orig_files))
    chile_highschool_gammaLarge_pred = pd.concat((pd.read_csv(file, sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"]) \
                                              for file in gammaLarge_pred_files))

    evaluate.protected_percentage_per_chunk_average_all_queries(chile_highschool_gammaLarge_pred, chunksize, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/UNI-Gamma_large-protected_percentage_per_chunk_highschool_nosemi.png');
    evaluate.evaluate(chile_highschool_gammaLarge_pred, chile_highschool_gammaLarge_orig, '../../octave-src/sample/ChileUni/NoSemi/highschool/results/evaluation_highschool_nosemi_gammaLarge.txt')


###############################################################################################
# TREC
###############################################################################################

if TREC:
    chunksize = 10

    # GAMMA 0
    trec_0_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_0_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(trec_0_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=0/TREC-Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_0_pred, trec_0_orig, '../../octave-src/sample/TREC/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA 15000
    trec_15000_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=15000/features_with_total_order-zscore-test.csv_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_15000_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=15000/features_with_total_order-zscore-test.csv_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(trec_15000_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=15000/TREC-Gamma_15000-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_15000_pred, trec_15000_orig, '../../octave-src/sample/TREC/GAMMA=15000/kendalls_tau_per_query.txt')


    # GAMMA 75000
    trec_75000_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=75000/features_with_total_order-zscore-test.csv_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_75000_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=75000/features_with_total_order-zscore-test.csv_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(trec_75000_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=75000/TREC-Gamma_75000-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_75000_pred, trec_75000_orig, '../../octave-src/sample/TREC/GAMMA=75000/kendalls_tau_per_query.txt')


    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_colorblind_orig = pd.read_csv('../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-test.csv_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_colorblind_pred = pd.read_csv('../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-test.csv_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_colorblind_orig, trec_colorblind_pred = add_prot_to_colorblind(orig_scores, trec_colorblind_orig, trec_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(trec_colorblind_pred, chunksize, '../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/TREC-Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_colorblind_pred, trec_colorblind_orig, '../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/kendalls_tau_per_query.txt')


###############################################################################################
# TREC BIG
###############################################################################################

if TREC_BIG:
    chunksize = 50

    # GAMMA 0
    trec_big_0_orig = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_big_0_pred = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(trec_big_0_pred, chunksize, '../../octave-src/sample/TREC-BIG/GAMMA=0/TREC-BIG-Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_big_0_pred, trec_big_0_orig, '../../octave-src/sample/TREC-BIG/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA SMALL
    trec_big_smallGamma_orig = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_big_smallGamma_pred = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=SMALL/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(trec_big_smallGamma_pred, chunksize, '../../octave-src/sample/TREC-BIG/GAMMA=SMALL/TREC-BIG-Gamma_SMALL-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_big_smallGamma_pred, trec_big_smallGamma_orig, '../../octave-src/sample/TREC-BIG/GAMMA=SMALL/kendalls_tau_per_query.txt')


    # GAMMA LARGE
    trec_big_largeGamma_orig = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_big_largeGamma_pred = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=LARGE/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(trec_big_largeGamma_pred, chunksize, '../../octave-src/sample/TREC-BIG/GAMMA=LARGE/TREC-BIG-Gamma_LARGE-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_big_largeGamma_pred, trec_big_largeGamma_orig, '../../octave-src/sample/TREC-BIG/GAMMA=LARGE/kendalls_tau_per_query.txt')

    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/TREC-BIG/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_big_colorblind_orig = pd.read_csv('../../octave-src/sample/TREC-BIG/COLORBLIND_GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_big_colorblind_pred = pd.read_csv('../../octave-src/sample/TREC-BIG/COLORBLIND_GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    trec_big_colorblind_orig, trec_big_colorblind_pred = add_prot_to_colorblind(orig_scores, trec_big_colorblind_orig, trec_big_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(trec_big_colorblind_pred, chunksize, '../../octave-src/sample/TREC-BIG/COLORBLIND_GAMMA=0/TREC-BIG-Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(trec_big_colorblind_pred, trec_big_colorblind_orig, '../../octave-src/sample/TREC-BIG/COLORBLIND_GAMMA=0/kendalls_tau_per_query.txt')


###############################################################################################
# LAW STUDENTS GENDER
###############################################################################################

if LAW_STUDENTS_GENDER:
    chunksize = 200

    # GAMMA 0
    lawStudents_gender_0_orig = pd.read_csv('../../octave-src/sample/LawStudents/gender/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_gender_0_pred = pd.read_csv('../../octave-src/sample/LawStudents/gender/GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_gender_0_pred, chunksize, '../../octave-src/sample/LawStudents/gender/GAMMA=0/LawStudents_Gender_Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_gender_0_pred, lawStudents_gender_0_orig, '../../octave-src/sample/LawStudents/gender/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA SMALL
    lawStudents_gender_smallGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/gender/GAMMA=SMALL/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_gender_smallGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/gender/GAMMA=SMALL/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_gender_smallGamma_pred, chunksize, '../../octave-src/sample/LawStudents/gender/GAMMA=SMALL/LawStudents_Gender_Gamma_SMALL-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_gender_smallGamma_pred, lawStudents_gender_smallGamma_orig, '../../octave-src/sample/LawStudents/gender/GAMMA=SMALL/kendalls_tau_per_query.txt')


    # GAMMA LARGE
    lawStudents_gender_largeGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/gender/GAMMA=LARGE/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_gender_largeGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/gender/GAMMA=LARGE/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_gender_largeGamma_pred, chunksize, '../../octave-src/sample/LawStudents/gender/GAMMA=LARGE/LawStudents_gender_Gamma_LARGE-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_gender_largeGamma_pred, lawStudents_gender_largeGamma_orig, '../../octave-src/sample/LawStudents/gender/GAMMA=LARGE/kendalls_tau_per_query.txt')


    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/LawStudents/gender/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_gender_colorblind_orig = pd.read_csv('../../octave-src/sample/LawStudents/gender/COLORBLIND/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_gender_colorblind_pred = pd.read_csv('../../octave-src/sample/LawStudents/gender/COLORBLIND/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_gender_colorblind_orig, lawStudents_gender_colorblind_pred = add_prot_to_colorblind(orig_scores, lawStudents_gender_colorblind_orig, lawStudents_gender_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_gender_colorblind_pred, chunksize, '../../octave-src/sample/LawStudents/gender/COLORBLIND/LawStudents_gender_Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_gender_colorblind_pred, lawStudents_gender_colorblind_orig, '../../octave-src/sample/LawStudents/gender/COLORBLIND/kendalls_tau_per_query.txt')


###############################################################################################
# LAW STUDENTS ASIAN
###############################################################################################

if LAW_STUDENTS_ASIAN:
    chunksize = 100

    # GAMMA 0
    lawStudents_asian_0_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_asian_0_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_asian_0_pred, chunksize, '../../octave-src/sample/LawStudents/race_asian/GAMMA=0/LawStudents_Asian_Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_asian_0_pred, lawStudents_asian_0_orig, '../../octave-src/sample/LawStudents/race_asian/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA SMALL
    lawStudents_asian_smallGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/GAMMA=SMALL/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_asian_smallGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/GAMMA=SMALL/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_asian_smallGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_asian/GAMMA=SMALL/LawStudents_Asian_Gamma_SMALL-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_asian_smallGamma_pred, lawStudents_asian_smallGamma_orig, '../../octave-src/sample/LawStudents/race_asian/GAMMA=SMALL/kendalls_tau_per_query.txt')


    # GAMMA LARGE
    lawStudents_asian_largeGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/GAMMA=LARGE/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_asian_largeGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/GAMMA=LARGE/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_asian_largeGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_asian/GAMMA=LARGE/LawStudents_asian_Gamma_LARGE-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_asian_largeGamma_pred, lawStudents_asian_largeGamma_orig, '../../octave-src/sample/LawStudents/race_asian/GAMMA=LARGE/kendalls_tau_per_query.txt')


    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_asian_colorblind_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/COLORBLIND/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_asian_colorblind_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_asian/COLORBLIND/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_asian_colorblind_orig, lawStudents_asian_colorblind_pred = add_prot_to_colorblind(orig_scores, lawStudents_asian_colorblind_orig, lawStudents_asian_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_asian_colorblind_pred, chunksize, '../../octave-src/sample/LawStudents/race_asian/COLORBLIND/LawStudents_asian_Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_asian_colorblind_pred, lawStudents_asian_colorblind_orig, '../../octave-src/sample/LawStudents/race_asian/COLORBLIND/kendalls_tau_per_query.txt')


###############################################################################################
# LAW STUDENTS BLACK
###############################################################################################

if LAW_STUDENTS_BLACK:
    chunksize = 100

    # GAMMA 0
    lawStudents_black_0_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_black/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_black_0_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_black/GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_black_0_pred, chunksize, '../../octave-src/sample/LawStudents/race_black/GAMMA=0/LawStudents_black_Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_black_0_pred, lawStudents_black_0_orig, '../../octave-src/sample/LawStudents/race_black/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA SMALL
    lawStudents_black_smallGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_black/GAMMA=SMALL/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_black_smallGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_black/GAMMA=SMALL/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_black_smallGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_black/GAMMA=SMALL/LawStudents_black_Gamma_SMALL-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_black_smallGamma_pred, lawStudents_black_smallGamma_orig, '../../octave-src/sample/LawStudents/race_black/GAMMA=SMALL/kendalls_tau_per_query.txt')


    # GAMMA LARGE
    lawStudents_black_largeGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_black/GAMMA=LARGE/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_black_largeGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_black/GAMMA=LARGE/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_black_largeGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_black/GAMMA=LARGE/LawStudents_black_Gamma_LARGE-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_black_largeGamma_pred, lawStudents_black_largeGamma_orig, '../../octave-src/sample/LawStudents/race_black/GAMMA=LARGE/kendalls_tau_per_query.txt')


    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/LawStudents/race_black/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_black_colorblind_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_black/COLORBLIND/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_black_colorblind_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_black/COLORBLIND/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_black_colorblind_orig, lawStudents_black_colorblind_pred = add_prot_to_colorblind(orig_scores, lawStudents_black_colorblind_orig, lawStudents_black_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_black_colorblind_pred, chunksize, '../../octave-src/sample/LawStudents/race_black/COLORBLIND/LawStudents_black_Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_black_colorblind_pred, lawStudents_black_colorblind_orig, '../../octave-src/sample/LawStudents/race_black/COLORBLIND/kendalls_tau_per_query.txt')


###############################################################################################
# LAW STUDENTS HISPANIC
###############################################################################################

if LAW_STUDENTS_HISPANIC:
    chunksize = 100

    # GAMMA 0
    lawStudents_hispanic_0_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_hispanic_0_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_hispanic_0_pred, chunksize, '../../octave-src/sample/LawStudents/race_hispanic/GAMMA=0/LawStudents_hispanic_Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_hispanic_0_pred, lawStudents_hispanic_0_orig, '../../octave-src/sample/LawStudents/race_hispanic/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA SMALL
    lawStudents_hispanic_smallGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/GAMMA=SMALL/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_hispanic_smallGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/GAMMA=SMALL/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_hispanic_smallGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_hispanic/GAMMA=SMALL/LawStudents_hispanic_Gamma_SMALL-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_hispanic_smallGamma_pred, lawStudents_hispanic_smallGamma_orig, '../../octave-src/sample/LawStudents/race_hispanic/GAMMA=SMALL/kendalls_tau_per_query.txt')


    # GAMMA LARGE
    lawStudents_hispanic_largeGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/GAMMA=LARGE/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_hispanic_largeGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/GAMMA=LARGE/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_hispanic_largeGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_hispanic/GAMMA=LARGE/LawStudents_hispanic_Gamma_LARGE-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_hispanic_largeGamma_pred, lawStudents_hispanic_largeGamma_orig, '../../octave-src/sample/LawStudents/race_hispanic/GAMMA=LARGE/kendalls_tau_per_query.txt')


    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_hispanic_colorblind_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/COLORBLIND/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_hispanic_colorblind_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_hispanic/COLORBLIND/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_hispanic_colorblind_orig, lawStudents_hispanic_colorblind_pred = add_prot_to_colorblind(orig_scores, lawStudents_hispanic_colorblind_orig, lawStudents_hispanic_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_hispanic_colorblind_pred, chunksize, '../../octave-src/sample/LawStudents/race_hispanic/COLORBLIND/LawStudents_hispanic_Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_hispanic_colorblind_pred, lawStudents_hispanic_colorblind_orig, '../../octave-src/sample/LawStudents/race_hispanic/COLORBLIND/kendalls_tau_per_query.txt')


###############################################################################################
# LAW STUDENTS MEXICAN
###############################################################################################

if LAW_STUDENTS_MEXICAN:
    chunksize = 100

    # GAMMA 0
    lawStudents_mexican_0_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_mexican_0_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_mexican_0_pred, chunksize, '../../octave-src/sample/LawStudents/race_mexican/GAMMA=0/LawStudents_mexican_Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_mexican_0_pred, lawStudents_mexican_0_orig, '../../octave-src/sample/LawStudents/race_mexican/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA SMALL
    lawStudents_mexican_smallGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/GAMMA=SMALL/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_mexican_smallGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/GAMMA=SMALL/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_mexican_smallGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_mexican/GAMMA=SMALL/LawStudents_mexican_Gamma_SMALL-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_mexican_smallGamma_pred, lawStudents_mexican_smallGamma_orig, '../../octave-src/sample/LawStudents/race_mexican/GAMMA=SMALL/kendalls_tau_per_query.txt')


    # GAMMA LARGE
    lawStudents_mexican_largeGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/GAMMA=LARGE/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_mexican_largeGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/GAMMA=LARGE/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_mexican_largeGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_mexican/GAMMA=LARGE/LawStudents_mexican_Gamma_LARGE-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_mexican_largeGamma_pred, lawStudents_mexican_largeGamma_orig, '../../octave-src/sample/LawStudents/race_mexican/GAMMA=LARGE/kendalls_tau_per_query.txt')


    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_mexican_colorblind_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/COLORBLIND/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_mexican_colorblind_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_mexican/COLORBLIND/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_mexican_colorblind_orig, lawStudents_mexican_colorblind_pred = add_prot_to_colorblind(orig_scores, lawStudents_mexican_colorblind_orig, lawStudents_mexican_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_mexican_colorblind_pred, chunksize, '../../octave-src/sample/LawStudents/race_mexican/COLORBLIND/LawStudents_mexican_Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_mexican_colorblind_pred, lawStudents_mexican_colorblind_orig, '../../octave-src/sample/LawStudents/race_mexican/COLORBLIND/kendalls_tau_per_query.txt')


###############################################################################################
# LAW STUDENTS PUERTORICAN
###############################################################################################

if LAW_STUDENTS_PUERTORICAN:
    chunksize = 100

    # GAMMA 0
    lawStudents_puertorican_0_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_puertorican_0_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/GAMMA=0/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_puertorican_0_pred, chunksize, '../../octave-src/sample/LawStudents/race_puertorican/GAMMA=0/LawStudents_puertorican_Gamma_0-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_puertorican_0_pred, lawStudents_puertorican_0_orig, '../../octave-src/sample/LawStudents/race_puertorican/GAMMA=0/kendalls_tau_per_query.txt')


    # GAMMA SMALL
    lawStudents_puertorican_smallGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/GAMMA=SMALL/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_puertorican_smallGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/GAMMA=SMALL/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_puertorican_smallGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_puertorican/GAMMA=SMALL/LawStudents_puertorican_Gamma_SMALL-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_puertorican_smallGamma_pred, lawStudents_puertorican_smallGamma_orig, '../../octave-src/sample/LawStudents/race_puertorican/GAMMA=SMALL/kendalls_tau_per_query.txt')


    # GAMMA LARGE
    lawStudents_puertorican_largeGamma_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/GAMMA=LARGE/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_puertorican_largeGamma_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/GAMMA=LARGE/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_puertorican_largeGamma_pred, chunksize, '../../octave-src/sample/LawStudents/race_puertorican/GAMMA=LARGE/LawStudents_puertorican_Gamma_LARGE-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_puertorican_largeGamma_pred, lawStudents_puertorican_largeGamma_orig, '../../octave-src/sample/LawStudents/race_puertorican/GAMMA=LARGE/kendalls_tau_per_query.txt')


    # COLORBLIND
    orig_scores = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/GAMMA=0/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_puertorican_colorblind_orig = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/COLORBLIND/trainingScores_ORIG.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_puertorican_colorblind_pred = pd.read_csv('../../octave-src/sample/LawStudents/race_puertorican/COLORBLIND/predictions_SORTED.pred',
                                      sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

    lawStudents_puertorican_colorblind_orig, lawStudents_puertorican_colorblind_pred = add_prot_to_colorblind(orig_scores, lawStudents_puertorican_colorblind_orig, lawStudents_puertorican_colorblind_pred)

    evaluate.protected_percentage_per_chunk_per_query(lawStudents_puertorican_colorblind_pred, chunksize, '../../octave-src/sample/LawStudents/race_puertorican/COLORBLIND/LawStudents_puertorican_Colorblind-protected_percentage_per_chunk_per_query.png')
    evaluate.evaluate(lawStudents_puertorican_colorblind_pred, lawStudents_puertorican_colorblind_orig, '../../octave-src/sample/LawStudents/race_puertorican/COLORBLIND/kendalls_tau_per_query.txt')



































