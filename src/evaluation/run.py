'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd

# call evaluate for all experiments (synthetic, chile, trec)
# save to separated files

#############################################################################################
# SYNTHETIC
#############################################################################################

Synthetic_maleTop_0_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt.GAMMA0_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Synthetic_maleTop_0_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt.GAMMA0_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

Synthetic_maleTop_500_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/sample_test_data_scoreAndGender_separated.txt.GAMMA500_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Synthetic_maleTop_500_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=500/sample_test_data_scoreAndGender_separated.txt.GAMMA500_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

Synthetic_maleTop_1000_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/sample_test_data_scoreAndGender_separated.txt.GAMMA1000_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Synthetic_maleTop_1000_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=1000/sample_test_data_scoreAndGender_separated.txt.GAMMA1000_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

Synthetic_maleTop_50000_unsort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/sample_test_data_scoreAndGender_separated.txt.GAMMA50000_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Synthetic_maleTop_50000_sort = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=50000/sample_test_data_scoreAndGender_separated.txt.GAMMA50000_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

###############################################################################################
# CHILE UNIVERSITY GENDER
###############################################################################################

Chile_gender_100000_unsort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_gender_test.txt.GAMMA100000_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Chile_gender_100000_sort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_gender_test.txt.GAMMA100000_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

###############################################################################################
# CHILE UNIVERSITY HIGHSCHOOL
###############################################################################################

Chile_gender_100000_unsort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_test.txt.GAMMA100000_UNSORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])
Chile_gender_100000_sort = pd.read_csv('../../octave-src/sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_test.txt.GAMMA100000_SORTED.pred',
                                  sep=",", names=["query_id", "doc_id", "prediction", "prot_attr"])

