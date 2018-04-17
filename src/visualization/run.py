'''
Created on Apr 17, 2018

@author: mzehlike
'''

import visualization.chileDatasetPreparation as prep
import pandas as pd
import visualization.heatmap as hm
import visualization.distribution as pdf

# plot heatmap for successful students
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
hm.cool_warm_heatmap(data, '../../data/ChileUniversity/heatmapSuccessfulStudents.png')

# plot heatmap for all students
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
hm.cool_warm_heatmap(data, '../../data/ChileUniversity/heatmapAllStudents.png')

############################################################################
# UNIVERSITY GRADES AFTER FIRST YEAR
############################################################################

# plot quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 3}
attributeQuality = "notas_"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_UniGrades_AllStudents.png',
         labels=['public', 'semi-private', 'private'])

attributeNamesAndCategories = {"highschool_type" : 3}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_UniGrades_SuccessfulStudents.png',
         labels=['public', 'semi-private', 'private'])


# plot quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_UniGrades_AllStudents.png',
         labels=['female', 'male'])

# plot quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_UniGrades_SuccessfulStudents.png',
         labels=['female', 'male'])

############################################################################
# HIGHSCHOOL FINAL GRADES
############################################################################

# plot quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 3}
attributeQuality = "nem"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_HighschoolGrades_AllStudents.png',
         labels=['public', 'semi-private', 'private'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_HighschoolGrades_SuccessfulStudents.png',
         labels=['public', 'semi-private', 'private'])


# plot quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_HighschoolGrades_AllStudents.png',
         labels=['female', 'male'])

# plot quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_HighschoolGrades_SuccessfulStudents.png',
         labels=['female', 'male'])


############################################################################
# PSU MATH SCORES
############################################################################

# plot quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 3}
attributeQuality = "psu_mat"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSUMath_AllStudents.png',
         labels=['public', 'semi-private', 'private'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_PSUMath_SuccessfulStudents.png',
         labels=['public', 'semi-private', 'private'])


# plot quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSUMath_AllStudents.png',
         labels=['female', 'male'])

# plot quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_PSUMath_SuccessfulStudents.png',
         labels=['female', 'male'])

############################################################################
# Fraction of Credits a Student failed
############################################################################

# plot quality distributions of university grades for different school types
attributeNamesAndCategories = {"highschool_type" : 3}
attributeQuality = "rat_ud"
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_FailRatio_AllStudents.png',
         labels=['public', 'semi-private', 'private'])

data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_HighschoolType_FailRatio_SuccessfulStudents.png',
         labels=['public', 'semi-private', 'private'])


# plot quality distributions of university grades for male and female
attributeNamesAndCategories = {"hombre" : 2}
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.allStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_FailRatio_AllStudents.png',
         labels=['female', 'male'])

# plot quality distributions of university grades for male and female
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.successfulStudents(data)
pdf.plot(data, attributeNamesAndCategories, attributeQuality,
         '../../data/ChileUniversity/pdf_Gender_FailRatio_SuccessfulStudents.png',
         labels=['female', 'male'])













