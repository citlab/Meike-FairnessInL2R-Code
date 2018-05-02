'''
Created on May 2, 2018

@author: mzehlike
'''
import pandas as pd
import visualization.distribution as pdf

attributeNamesAndCategories = {"gender" : 2}
attributeQuality = "score"

data = pd.read_csv('../../octave-src/sample/synthetic_score_gender/distribution_based/sample_train_data_scoreAndGender.txt',
                   sep=",", names=["query_id", "gender", "score", "rank"])

pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_HighschoolGrades_SuccessfulStudents.png',
         labels=['female', 'male'])
