'''
Created on May 2, 2018

@author: mzehlike
'''
import pandas as pd
import visualization.distribution as pdf
import visualization.plot_rankings_only_protection_status as pltres

attributeNamesAndCategories = {"gender" : 2}
attributeQuality = "score"

k = 100

############################################################################
# NORMAL DISTRIBUTION TOP MALE BOTTOM FEMALE
############################################################################

# DATASET PLOT
data = pd.read_csv('../../octave-src/sample/synthetic_score_gender/distribution_based/sample_train_data_scoreAndGender_normalDistribution.txt',
                   sep=",", names=["query_id", "gender", "score", "rank"])

pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../plots/synthetic/separated/top_male_bottom_female/dataset_plots/normal_distribution/pdf_scores.png',
         labels=['non-protected', 'protected'])

# RESULT PLOT
input_file1 = '../../octave-src/sample/synthetic_score_gender/distribution_based/sample_train_data_scoreAndGender_normalDistribution.txt'
input_file2 = '../../octave-src/sample/synthetic_score_gender/distribution_based/sample_test_data_scoreAndGender_normalDistribution_GAMMA_ZERO.txt.pred'
input_file3 = '../../octave-src/sample/synthetic_score_gender/distribution_based/sample_test_data_scoreAndGender_normalDistribution_GAMMA_MEDIUM.txt.pred'
input_file4 = '../../octave-src/sample/synthetic_score_gender/distribution_based/sample_test_data_scoreAndGender_normalDistribution_GAMMA_LARGE.txt.pred'
output_file = '../../plots/synthetic/separated/top_male_bottom_female/result_plots/normal_distribution/rankings_group_property_only.png'

pltres.plot_rankings(input_file1, input_file2, input_file3, input_file4, output_file, 500, 10)
