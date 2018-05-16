'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd
from evaluation import evaluate


#############################################################################################
# SYNTHETIC
#############################################################################################

chunksize = 5

# GAMMA 0

synthetic_male_top_0_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
synthetic_male_top_0_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(synthetic_male_top_0_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/protected_percentage_per_chunk.png');
evaluate.evaluate(synthetic_male_top_0_pred,
                  synthetic_male_top_0_orig,
                  '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/kendalls_tau_per_query.txt',
                  synthetic=True)

# GAMMA 75

synthetic_male_top_75_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
synthetic_male_top_75_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(synthetic_male_top_75_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/protected_percentage_per_chunk.png');
evaluate.evaluate(synthetic_male_top_75_pred,
                  synthetic_male_top_75_orig,
                  '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=75/kendalls_tau_per_query.txt',
                  synthetic=True)

# GAMMA 150

synthetic_male_top_150_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
synthetic_male_top_150_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(synthetic_male_top_150_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/protected_percentage_per_chunk.png');
evaluate.evaluate(synthetic_male_top_150_pred,
                  synthetic_male_top_150_orig,
                  '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=150/kendalls_tau_per_query.txt',
                  synthetic=True)


###############################################################################################
# CHILE UNIVERSITY GENDER WITH SEMI-PRIVATE
###############################################################################################

def add_prot_to_colorblind(orig_scores, colorblind_orig, colorblind_pred):

    orig_prot_attr = orig_scores['prot_attr']
    colorblind_orig["prot_attr"] = orig_prot_attr

    for doc_id in colorblind_orig['doc_id']:
        prot_status_for_pred = colorblind_orig.loc[colorblind_orig['doc_id'] == doc_id]['prot_attr'].values
        colorblind_pred.at[colorblind_pred['doc_id'] == doc_id, 'prot_attr'] = prot_status_for_pred

    return colorblind_orig, colorblind_pred

chunksize = 30

# COLORBLIND TRAINING
orig_scores = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_gender_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_gender_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_colorblind_gender_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_gender_orig, chile_colorblind_gender_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_gender_orig, chile_colorblind_gender_pred)

evaluate.protected_percentage_per_chunk(chile_colorblind_gender_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/UNI-Colorblind-protected_percentage_per_chunk_gender.png');
evaluate.evaluate(chile_colorblind_gender_pred, chile_colorblind_gender_orig, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/kendalls_tau_per_query_gender.txt')

# GAMMA 0
chile_gender_0_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_gender_test.txt_ORIG.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_gender_0_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_gender_test.txt_SORTED.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_gender_0_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/UNI-Gamma_0-protected_percentage_per_chunk_gender.png');
evaluate.evaluate(chile_gender_0_pred, chile_gender_0_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/kendalls_tau_per_query_gender.txt')


# GAMMA 100000
chile_gender_100000_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_gender_test.txt_ORIG.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_gender_100000_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_gender_test.txt_SORTED.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_gender_100000_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/UNI-Gamma_100000-protected_percentage_per_chunk_gender.png');
evaluate.evaluate(chile_gender_100000_pred, chile_gender_100000_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/kendalls_tau_per_query_gender.txt')

# GAMMA 5000000
chile_gender_5000000_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_gender_test.txt_ORIG.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_gender_5000000_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_gender_test.txt_SORTED.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_gender_5000000_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/UNI-Gamma_5000000-protected_percentage_per_chunk_gender.png');
evaluate.evaluate(chile_gender_5000000_pred, chile_gender_5000000_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/kendalls_tau_per_query_gender.txt')

###############################################################################################
# CHILE UNIVERSITY HIGHSCHOOL WITH SEMI-PRIVATE
###############################################################################################

chunksize = 30

# COLORBLIND TRAINING
orig_scores = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_highschool_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_highschool_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_colorblind_highschool_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_highschool_orig, chile_colorblind_highschool_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_highschool_orig, chile_colorblind_highschool_pred)

evaluate.protected_percentage_per_chunk(chile_colorblind_highschool_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/UNI-Colorblind-protected_percentage_per_chunk_highschool.png');
evaluate.evaluate(chile_colorblind_highschool_pred, chile_colorblind_highschool_orig, '../../octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/kendalls_tau_per_query_highschool.txt')

# GAMMA 0
chile_highschool_0_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_highschool_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_highschool_0_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=0/chileDataL2R_highschool_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_highschool_0_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/UNI-Gamma_0-protected_percentage_per_chunk_highschool.png');
evaluate.evaluate(chile_highschool_0_pred, chile_highschool_0_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=0/kendalls_tau_per_query_highschool.txt')


# GAMMA 100000
chile_highschool_100000_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_highschool_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_highschool_100000_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=100000/chileDataL2R_highschool_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_highschool_100000_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/UNI-Gamma_100000-protected_percentage_per_chunk_highschool.png');
evaluate.evaluate(chile_highschool_100000_pred, chile_highschool_100000_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=100000/kendalls_tau_per_query_highschool.txt')

# GAMMA 5000000
chile_highschool_5000000_orig = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_highschool_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_highschool_5000000_pred = pd.read_csv('../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/chileDataL2R_highschool_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_highschool_5000000_pred, chunksize, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/UNI-Gamma_5000000-protected_percentage_per_chunk_highschool.png');
evaluate.evaluate(chile_highschool_5000000_pred, chile_highschool_5000000_orig, '../../octave-src/sample/ChileUni/Semi/GAMMA=5000000/kendalls_tau_per_query_highschool.txt')

###############################################################################################
# TREC
###############################################################################################

chunksize = 10

# GAMMA 0
trec_0_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_0_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(trec_0_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=0/TREC-Gamma_0-protected_percentage_per_chunk.png')
evaluate.evaluate(trec_0_pred, trec_0_orig, '../../octave-src/sample/TREC/GAMMA=0/kendalls_tau_per_query.txt')


# GAMMA 15000
trec_15000_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=15000/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_15000_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=15000/features_with_total_order-zscore-test.csv_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(trec_15000_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=15000/TREC-Gamma_15000-protected_percentage_per_chunk.png')
evaluate.evaluate(trec_15000_pred, trec_15000_orig, '../../octave-src/sample/TREC/GAMMA=15000/kendalls_tau_per_query.txt')


# GAMMA 75000
trec_75000_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=75000/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_75000_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=75000/features_with_total_order-zscore-test.csv_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(trec_75000_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=75000/TREC-Gamma_75000-protected_percentage_per_chunk.png')
evaluate.evaluate(trec_75000_pred, trec_75000_orig, '../../octave-src/sample/TREC/GAMMA=75000/kendalls_tau_per_query.txt')


# COLORBLIND
orig_scores = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_colorblind_orig = pd.read_csv('../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_colorblind_pred = pd.read_csv('../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-test.csv_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_colorblind_orig, trec_colorblind_pred = add_prot_to_colorblind(orig_scores, trec_colorblind_orig, trec_colorblind_pred)

evaluate.protected_percentage_per_chunk(trec_colorblind_pred, chunksize, '../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/TREC-Colorblind-protected_percentage_per_chunk.png')
evaluate.evaluate(trec_colorblind_pred, trec_colorblind_orig, '../../octave-src/sample/TREC/COLORBLIND_GAMMA=0/kendalls_tau_per_query.txt')
















