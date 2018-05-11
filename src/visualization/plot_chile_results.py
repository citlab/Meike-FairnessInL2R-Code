'''
Created on May 10, 2018

@author: mzehlike
'''
import visualization.ranking_results_only_protection_status as rankres

attributeNamesAndCategories = {"gender" : 2}
attributeQuality = "score"


###########################################################################
# CHILE GENDER
###########################################################################

# RESULT PLOT
input_file1 = '../../data/ChileUniversity/chileDataL2R_gender_test.txt'
input_file2 = '../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_test_data_scoreAndGender_separated_GAMMA_ZERO.txt.pred'
input_file3 = '../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_test_data_scoreAndGender_separated_GAMMA_MEDIUM.txt.pred'
input_file4 = '../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_test_data_scoreAndGender_separated_GAMMA_LARGE.txt.pred'
output_file = '../../plots/synthetic/separated/top_male_bottom_female/result_plots/uniform_distribution/uniform_male_top_rankings_group_property_only.png'

rankres.plot_rankings(input_file1, input_file2, input_file3, input_file4, output_file, 50, 1)