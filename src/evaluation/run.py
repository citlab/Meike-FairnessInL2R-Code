'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd
from evaluation import evaluate

# call evaluate for all experiments (synthetic, chile, trec)
# save to separated files

#############################################################################################
# SYNTHETIC
#############################################################################################

# GAMMA 0

Synthetic_maleTop_0_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Synthetic_maleTop_0_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

# evaluate.calculate_protected_percentage_per_chunk(Synthetic_maleTop_0_sort, 5, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/protected_percentage_per_chunk.png');
# evaluate.calculate_kendalls_tau(Synthetic_maleTop_0_sort, Synthetic_maleTop_0_unsort)

# GAMMA 500

Synthetic_maleTop_500_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/sample_test_data_scoreAndGender_separated.txt_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Synthetic_maleTop_500_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

# evaluate.calculate_protected_percentage_per_chunk(Synthetic_maleTop_500_sort, 5, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/protected_percentage_per_chunk.png');
evaluate.calculate_kendalls_tau(Synthetic_maleTop_500_sort, Synthetic_maleTop_500_unsort, '../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/kendalls_tau_per_query.txt')

# GAMMA 1000

# Synthetic_maleTop_1000_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/sample_test_data_scoreAndGender_separated.txt_UNSORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
# Synthetic_maleTop_1000_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
# # GAMMA 50000
#
# Synthetic_maleTop_50000_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/sample_test_data_scoreAndGender_separated.txt_UNSORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
# Synthetic_maleTop_50000_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/sample_test_data_scoreAndGender_separated.txt_SORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
# ###############################################################################################
# # CHILE UNIVERSITY GENDER
# ###############################################################################################
#
# Chile_gender_100000_unsort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_gender_test.txt_UNSORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
# Chile_gender_100000_sort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_gender_test.txt_SORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
#
# evaluate.calculate_protected_percentage_per_chunk(Chile_gender_100000_sort, 50, '../../octave-src/sample/ChileUni/GAMMA=100000/protected_percentage_per_chunk.png');
#
#
# ###############################################################################################
# # CHILE UNIVERSITY HIGHSCHOOL
# ###############################################################################################
#
# Chile_highschool_100000_unsort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_test.txt_UNSORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
# Chile_highschool_100000_sort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_test.txt_SORTED.pred',
#                                   sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

