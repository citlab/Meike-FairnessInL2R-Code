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
evaluate.evaluate(synthetic_male_top_0_pred, synthetic_male_top_0_orig, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/kendalls_tau_per_query.txt')

# GAMMA 500

synthetic_male_top_500_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
synthetic_male_top_500_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(synthetic_male_top_500_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/protected_percentage_per_chunk.png');
evaluate.evaluate(synthetic_male_top_500_pred, synthetic_male_top_500_orig, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/kendalls_tau_per_query.txt')

# GAMMA 1000

synthetic_male_top_1000_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
synthetic_male_top_1000_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(synthetic_male_top_1000_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/protected_percentage_per_chunk.png');
evaluate.evaluate(synthetic_male_top_1000_pred, synthetic_male_top_1000_orig, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/kendalls_tau_per_query.txt')


# GAMMA 50000

synthetic_male_top_50000_orig = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/sample_test_data_scoreAndGender_separated.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
synthetic_male_top_50000_pred = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(synthetic_male_top_50000_pred, chunksize, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/protected_percentage_per_chunk.png');
evaluate.evaluate(synthetic_male_top_500_pred, synthetic_male_top_50000_orig, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/kendalls_tau_per_query.txt')


###############################################################################################
# CHILE UNIVERSITY GENDER
###############################################################################################

def add_prot_to_colorblind(orig_scores, colorblind_orig, colorblind_pred):

    orig_prot_attr = orig_scores['prot_attr']
    colorblind_orig["prot_attr"] = orig_prot_attr

    for doc_id in colorblind_orig['doc_id']:
        prot_status_for_pred = colorblind_orig.loc[colorblind_orig['doc_id'] == doc_id]['prot_attr'].values
        colorblind_pred.at[colorblind_pred['doc_id'] == doc_id, 'prot_attr'] = prot_status_for_pred

    return colorblind_orig, colorblind_pred

chunksize = 50

# COLORBLIND TRAINING
orig_scores = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=0/chileDataL2R_gender_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_gender_orig = pd.read_csv('../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_colorblind_gender_pred = pd.read_csv('../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_gender_orig, chile_colorblind_gender_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_gender_orig, chile_colorblind_gender_pred)

evaluate.protected_percentage_per_chunk(chile_colorblind_gender_pred, chunksize, '../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/protected_percentage_per_chunk_gender.png');
evaluate.evaluate(chile_colorblind_gender_pred, chile_colorblind_gender_orig, '../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/kendalls_tau_per_query_gender.txt')

# GAMMA 0
chile_gender_0_orig = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=0/chileDataL2R_gender_test.txt_ORIG.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_gender_0_pred = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=0/chileDataL2R_gender_test.txt_SORTED.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_gender_0_pred, chunksize, '../../octave-src/sample/ChileUni/GAMMA=0/protected_percentage_per_chunk_gender.png');
evaluate.evaluate(chile_gender_0_pred, chile_gender_0_orig, '../../octave-src/sample/ChileUni/GAMMA=0/kendalls_tau_per_query_gender.txt')


# GAMMA 100000
chile_gender_100000_orig = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_gender_test.txt_ORIG.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_gender_100000_pred = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_gender_test.txt_SORTED.pred',
                                sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_gender_100000_pred, chunksize, '../../octave-src/sample/ChileUni/GAMMA=100000/protected_percentage_per_chunk_gender.png');
evaluate.evaluate(chile_gender_100000_pred, chile_gender_100000_orig, '../../octave-src/sample/ChileUni/GAMMA=100000/kendalls_tau_per_query_gender.txt')

# GAMMA 5000000


###############################################################################################
# CHILE UNIVERSITY HIGHSCHOOL
###############################################################################################

chunksize = 50

# COLORBLIND TRAINING
orig_scores = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=0/chileDataL2R_highschool_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_highschool_orig = pd.read_csv('../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_colorblind_highschool_pred = pd.read_csv('../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

chile_colorblind_highschool_orig, chile_colorblind_highschool_pred = add_prot_to_colorblind(orig_scores, chile_colorblind_highschool_orig, chile_colorblind_highschool_pred)

evaluate.protected_percentage_per_chunk(chile_colorblind_highschool_pred, chunksize, '../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/protected_percentage_per_chunk_highschool.png');
evaluate.evaluate(chile_colorblind_highschool_pred, chile_colorblind_highschool_orig, '../../octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/kendalls_tau_per_query_highschool.txt')

# GAMMA 0
chile_highschool_0_orig = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=0/chileDataL2R_highschool_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_highschool_0_pred = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=0/chileDataL2R_highschool_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_highschool_0_pred, chunksize, '../../octave-src/sample/ChileUni/GAMMA=0/protected_percentage_per_chunk_highschool.png');
evaluate.evaluate(chile_highschool_0_pred, chile_highschool_0_orig, '../../octave-src/sample/ChileUni/GAMMA=0/kendalls_tau_per_query_highschool.txt')


# GAMMA 100000
chile_highschool_100000_orig = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_test.txt_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
chile_highschool_100000_pred = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_test.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(chile_highschool_100000_pred, chunksize, '../../octave-src/sample/ChileUni/GAMMA=100000/protected_percentage_per_chunk_highschool.png');
evaluate.evaluate(chile_highschool_100000_pred, chile_highschool_100000_orig, '../../octave-src/sample/ChileUni/GAMMA=100000/kendalls_tau_per_query_highschool.txt')

# GAMMA 5000000

###############################################################################################
# TREC
###############################################################################################

chunksize = 20

# GAMMA 0
trec_0_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_0_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(trec_0_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=0/protected_percentage_per_chunk.png')
evaluate.evaluate(trec_0_pred, trec_0_orig, '../../octave-src/sample/TREC/GAMMA=0/kendalls_tau_per_query.txt')


# GAMMA 10000
trec_10000_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=10000/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_10000_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=10000/features_with_total_order-zscore-test.csv_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(trec_10000_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=10000/protected_percentage_per_chunk.png')
evaluate.evaluate(trec_10000_pred, trec_10000_orig, '../../octave-src/sample/TREC/GAMMA=10000/kendalls_tau_per_query.txt')


# GAMMA 500000
trec_500000_orig = pd.read_csv('../../octave-src/sample/TREC/GAMMA=500000/features_with_total_order-zscore-test.csv_ORIG.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

trec_500000_pred = pd.read_csv('../../octave-src/sample/TREC/GAMMA=500000/features_with_total_order-zscore-test.csv_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

evaluate.protected_percentage_per_chunk(trec_500000_pred, chunksize, '../../octave-src/sample/TREC/GAMMA=500000/protected_percentage_per_chunk.png')
evaluate.evaluate(trec_500000_pred, trec_500000_orig, '../../octave-src/sample/TREC/GAMMA=500000/kendalls_tau_per_query.txt')









