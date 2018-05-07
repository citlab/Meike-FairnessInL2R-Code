'''
Created on May 7, 2018

@author: mzehlike
'''

import pandas as pd
import util.chileDatasetPreparation as prep

# write dataset for gender
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.prepareForL2R(data)
data.to_csv('../../data/ChileUniversity/chileDataL2R_gender.txt', index=False, header=False)

# write dataset for highschool type
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.prepareForL2R(data, gender=False)
data.to_csv('../../data/ChileUniversity/chileDataL2R_highschool.txt', index=False, header=False)