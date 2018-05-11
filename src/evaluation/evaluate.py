'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd

##################################################################################
# EVALUATE PRECENTAGE OF PROTECTED CANDIDATES
##################################################################################

Synthetic_maleTop_0 = pd.read_csv('../../octave-src/sample/synthetic/top_male_bottom_female/GAMMA=0/',
                   sep=",", names=["query_id", "gender", "score", "rank"])

##################################################################################
# EVALUATE EXPOSURE DIFFERENCE BETWEEN GROUPS
##################################################################################

##################################################################################
# EVALUATE KENDALL'S TAU
##################################################################################


def protected_percentage(ranking):
    '''
    evaluates the percentage of protected candidates in the top 5% 10%, 25%, 50% percent of the
    given ranking
    '''
    top5 = 0;
    top10 = 0;
    top25 = 0;
    top50 = 0;

    return top5, top10, top25, top50
