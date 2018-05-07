'''
Created on Apr 17, 2018

@author: mzehlike
'''

import util.chileDatasetPreparation as prep
import pandas as pd
import visualization.heatmap as hm
import visualization.distribution as pdf

# plot_ChileDataset heatmap for successful students
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
hm.cool_warm_heatmap(data, '../../data/ChileUniversity/heatmapSuccessfulStudents.png')

# plot_ChileDataset heatmap for all students
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
hm.cool_warm_heatmap(data, '../../data/ChileUniversity/heatmapAllStudents.png')

############################################################################
# UNIVERSITY GRADES AFTER FIRST YEAR
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "notas_"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_UniGrades_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_UniGrades_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_UniGrades_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_UniGrades_SuccessfulStudents.png',
         labels=['male', 'female'])

############################################################################
# HIGHSCHOOL FINAL GRADES
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "nem"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_HighschoolGrades_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_HighschoolGrades_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_HighschoolGrades_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_HighschoolGrades_SuccessfulStudents.png',
         labels=['male', 'female'])


############################################################################
# PSU MATH SCORES
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "psu_mat"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSUMath_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSUMath_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSUMath_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSUMath_SuccessfulStudents.png',
         labels=['male', 'female'])


############################################################################
# PSU LANGUAGE SCORES
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "psu_len"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSULang_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSULang_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSULang_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSULang_SuccessfulStudents.png',
         labels=['male', 'female'])

############################################################################
# Fraction of Credits a Student failed
############################################################################

# plot_ChileDataset quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 2}
attributeQuality = "rat_ud"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_FailRatio_AllStudents.png',
         labels=['non-public', 'public'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_FailRatio_SuccessfulStudents.png',
         labels=['non-public', 'public'])


# plot_ChileDataset quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_FailRatio_AllStudents.png',
         labels=['male', 'female'])

# plot_ChileDataset quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot_ChileDataset(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_FailRatio_SuccessfulStudents.png',
         labels=['male', 'female'])


##################################################################################################
# PLOT RESULT GRAPHS
##################################################################################################










