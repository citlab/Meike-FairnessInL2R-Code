'''
Created on Apr 17, 2018

@author: mzehlike
'''

import util.chileDatasetPreparation as prep
import pandas as pd

##############################################################################################
# BOXPLOTS
##############################################################################################

# write dataset for gender
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
train, test = prep.prepareForL2R(data)

data = pd.concat(train, test)











