'''
Created on May 28, 2018

@author: mzehlike

protected attributes: sex, race
features: Law School Admission Test (LSAT), grade point average (UGPA)

training judgments: first year average grade (ZFYA)

excluding for now: region-first, sander-index, first_pf


höchste ID: 27476

Aufteilung in Trainings und Testdaten, 80% Training, 20% Testing, Random Sampling
'''

import pandas as pd
from scipy.stats import stats


def prepareGenderData():
    data = pd.read_excel('../../octave-src/sample/LawStudents/law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf', 'race'])

    data['sex'] = data['sex'].replace([2], 0)

    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['sex', 'LSAT', 'UGPA', 'ZFYA']]
    data.insert(0, 'query_dummy', 1)

    train = data.sample(frac=.8)
    test = data.drop(train.index)

#     print(train.shape)
#     print(test.shape)

    return train, test


def prepareRaceData(protGroup, nonprotGroup):
    data = pd.read_excel('../../octave-src/sample/LawStudents/law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf', 'sex'])

    data['race'] = data['race'].replace(to_replace=protGroup, value=1)
    data['race'] = data['race'].replace(to_replace=nonprotGroup, value=0)

    data = data[data['race'].isin([0, 1])]

    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['race', 'LSAT', 'UGPA', 'ZFYA']]
    data.insert(0, 'query_dummy', 1)

    train = data.sample(frac=.8)
    test = data.drop(train.index)

#     print(train.shape)
#     print(test.shape)

    return train, test


def prepareAllInOneDataForFAIR():
    data = pd.read_excel('../../octave-src/sample/LawStudents/law_data.csv.xlsx')
    data = data.drop(columns=['region_first', 'sander_index', 'first_pf'])

    data['sex'] = data['sex'].replace([2], 0)

    data['race'] = data['race'].replace(to_replace="White", value=0)
    data['race'] = data['race'].replace(to_replace="Amerindian", value=1)
    data['race'] = data['race'].replace(to_replace="Asian", value=2)
    data['race'] = data['race'].replace(to_replace="Black", value=3)
    data['race'] = data['race'].replace(to_replace="Hispanic", value=4)
    data['race'] = data['race'].replace(to_replace="Mexican", value=5)
    data['race'] = data['race'].replace(to_replace="Other", value=6)
    data['race'] = data['race'].replace(to_replace="Puertorican", value=7)

    data['LSAT'] = stats.zscore(data['LSAT'])
    data['UGPA'] = stats.zscore(data['UGPA'])

    data = data[['sex', 'race', 'LSAT', 'UGPA', 'ZFYA']]

    return data


######################################################################################
# GENDER
######################################################################################
train, test = prepareGenderData()
train.to_csv('../../octave-src/sample/LawStudents/gender/LawStudents_Gender_train.txt', index=False, header=False)
test.to_csv('../../octave-src/sample/LawStudents/gender/LawStudents_Gender_test.txt', index=False, header=False)

######################################################################################
# RACE
######################################################################################

train, test = prepareRaceData('Asian', 'White')
train.to_csv('../../octave-src/sample/LawStudents/race_asian/LawStudents_Race_train.txt', index=False, header=False)
test.to_csv('../../octave-src/sample/LawStudents/race_asian/LawStudents_Race_test.txt', index=False, header=False)

train, test = prepareRaceData('Black', 'White')
train.to_csv('../../octave-src/sample/LawStudents/race_black/LawStudents_Race_train.txt', index=False, header=False)
test.to_csv('../../octave-src/sample/LawStudents/race_black/LawStudents_Race_test.txt', index=False, header=False)

train, test = prepareRaceData('Hispanic', 'White')
train.to_csv('../../octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_train.txt', index=False, header=False)
test.to_csv('../../octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_test.txt', index=False, header=False)

train, test = prepareRaceData('Mexican', 'White')
train.to_csv('../../octave-src/sample/LawStudents/race_mexican/LawStudents_Race_train.txt', index=False, header=False)
test.to_csv('../../octave-src/sample/LawStudents/race_mexican/LawStudents_Race_test.txt', index=False, header=False)

train, test = prepareRaceData('Puertorican', 'White')
train.to_csv('../../octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_train.txt', index=False, header=False)
test.to_csv('../../octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_test.txt', index=False, header=False)

#######################################################################################
# ALL IN ONE
#######################################################################################

data = prepareAllInOneDataForFAIR()
data.to_csv('../../data/LSAT/LSAT_AllInOne.csv', index=False, header=True)












