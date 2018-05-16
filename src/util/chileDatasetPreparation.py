'''
Created on Apr 17, 2018

@author: mzehlike
'''
import numpy as np
import scipy.stats as stats
from conda._vendor.auxlib.collection import first


def principalDataPreparation_withSemiPrivate(data):
    # drop all uninteresting columns
    data = data.drop(columns=['rank_in', 'rank_em', 'sipee', 'bea', 'deporte', 'genero'])
    data = data.fillna(value={'uds_e_':0})  # fill NaNs with 0 in this specific column only

    # take only those students that have been admitted through the PSU tests, not by other things
    # data = data[data['tip_ing'] == 'PAA o PSU']

    # merge muni, sub and part column into one highschool type column
    # muni = 0, sub & part = 1, part = 2
    data['muni'] = data['muni'].replace([0], np.nan)
    data['muni'] = data['muni'].replace([1], 0)
    data['sub'] = data['sub'].replace([0], np.nan)
    data['part'] = data['part'].replace([0], np.nan)
    data['part'] = data['part'].replace([1], 1)
    data['muni'] = data['muni'].fillna(data['part'])
    data['muni'] = data['muni'].fillna(data['sub'])
    data = data.rename(index=str, columns={"muni":"highschool_type"})
    data = data.drop(columns=['sub', 'part'])
    # drop remaining nans
    data = data.dropna(subset=['highschool_type'])

    # switch protected attribute to be 1
    data['hombre'] = data['hombre'].replace({0:1, 1:0})
    data['highschool_type'] = data['highschool_type'].replace({0:1, 1:0})


    return data


def principalDataPreparation_withoutSemiPrivate(data):
    # drop all uninteresting columns
    data = data.drop(columns=['rank_in', 'rank_em', 'sipee', 'bea', 'deporte', 'genero'])
    data = data.fillna(value={'uds_e_':0})  # fill NaNs with 0 in this specific column only

    # merge muni, sub and part column into one highschool type column
    # muni = 0, sub & part = 1, part = 2
    data['muni'] = data['muni'].replace([0], np.nan)
    data['muni'] = data['muni'].replace([1], 0)
    data['part'] = data['part'].replace([0], np.nan)
    data['part'] = data['part'].replace([1], 1)
    data['muni'] = data['muni'].fillna(data['part'])
    data = data.rename(index=str, columns={"muni":"highschool_type"})
    data = data.drop(columns=['sub', 'part'])
    # drop remaining nans
    data = data.dropna(subset=['highschool_type'])

    # switch protected attribute to be 1
    data['hombre'] = data['hombre'].replace({0:1, 1:0})
    data['highschool_type'] = data['highschool_type'].replace({0:1, 1:0})


    return data



def successfulStudents(data):
    """
    this heatmap considers only students that have succeeded in the first semester of university
    success means that they took at least 45 credits and failed (or dropped) at most 10 of them
    """

    data = data[data['sem'] == 1]
    data = data[data['inactivo'] != 1]
    data = data[data['uds_e_'] <= 10]
    # after the above steps these columns contain all the same value and hence can be dropped
    data = data.drop(columns=['sem', 'inactivo', 'uds_e_'])
    data = data.dropna(subset=['nem', 'psu_mat', 'psu_len', 'psu_cie', 'psu_pond', 'notas_', 'rat_ud'])
    data = data[data['uds_i_'] >= 45]
    data = data[data['uds_r_'] <= 10]

    return data



def allStudents(data):
    """
    this heatmap considers all students that have taken exams after the first semester of university
    this means they were not "inactive" and non of their values in the relevant columns were NaN
    """

    data = data[data['sem'] == 1]
    data = data[data['inactivo'] != 1]
        # after the above steps these columns contain all the same value and hence can be dropped
    data = data.drop(columns=['sem', 'inactivo'])
    data = data.dropna(subset=['nem', 'psu_mat', 'psu_len', 'psu_cie', 'psu_pond', 'notas_', 'uds_i_',
                               'uds_r_', 'uds_e_', 'rat_ud'])

    return data

def prepareForL2R(data, gender=True, colorblind=False):
    """
    brings data into the correct format for L2R octave code with following scheme
    query_id; protection_status; feature_1; ...; feature_n; rank

    in this particular case we use year of university entrance as query_id, psu scores as features and notas as rank

    writes one dataset with protection_status "gender" and one with protection_status "highschool_type"
    """

    def rank(x, sortby):
        x.sort_values([sortby], ascending=False, inplace=True)
        return x

    data = data[data['sem'] == 1]
    data = data[data['inactivo'] != 1]

    # drop all lines where values are missing
    data = data.dropna(subset=['nem', 'psu_mat', 'psu_len', 'psu_cie', 'notas_', 'uds_i_'])

    print(data['hombre'].value_counts())
    print(data['highschool_type'].value_counts())

    # drop all columns that are not needed
    if(gender):
        keep_cols = ['ano_in', 'hombre', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'notas_', 'uds_i_', 'uds_r_', 'uds_e_']
    else:
        keep_cols = ['ano_in', 'highschool_type', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'notas_', 'uds_i_', 'uds_r_', 'uds_e_']

    if(colorblind):
        keep_cols = ['ano_in', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'notas_', 'uds_i_', 'uds_r_', 'uds_e_']

    data = data[keep_cols]

    # replace NaNs with zeros
    data['uds_r_'].fillna(0)
    data['uds_e_'].fillna(0)

    # add new column for ranking scores
    data['score'] = np.zeros(data.shape[0])

    # calculate score based on grades and credits
    for idx, row in data.iterrows():
        grades = row.loc['notas_']
        credits_taken = row.loc['uds_i_']
        credits_failed = row.loc['uds_r_']
        credits_dropped = row.loc['uds_e_']

        score = grades * (credits_taken - credits_failed - credits_dropped) / credits_taken
        data.loc[idx, 'score'] = score

    # group years together and rank them by notas
    data = data.groupby(data['ano_in'], as_index=False, sort=False).apply(rank, ('score'))

    # drop all columns that were used to calculate ranks, so that they wouldn't correlate on default
    data = data.drop(columns=['notas_', 'uds_i_', 'uds_r_', 'uds_e_'])

    # zscore psu scores and normalize scores
    def apply_zscores(x):
        x['psu_mat'] = stats.zscore(x['psu_mat'])
        x['psu_len'] = stats.zscore(x['psu_len'])
        x['psu_cie'] = stats.zscore(x['psu_cie'])
        x['nem'] = stats.zscore(x['nem'])
        x['score'] = stats.zscore(x['score'])
        return x

    data = data.groupby(data['ano_in'], as_index=False, sort=False).apply(lambda x: apply_zscores(x))

    # data[['psu_mat', 'psu_len', 'psu_cie', 'nem']] = data.groupby(data['ano_in'], as_index=False, sort=False)[['psu_mat', 'psu_len', 'psu_cie', 'nem']].transform(lambda x: x / x.max())

    # replace ano_in with query_ids that start from 1
    data['ano_in'] = data['ano_in'].replace(to_replace=2010, value=1)
    data['ano_in'] = data['ano_in'].replace(to_replace=2011, value=2)
    data['ano_in'] = data['ano_in'].replace(to_replace=2012, value=3)
    data['ano_in'] = data['ano_in'].replace(to_replace=2013, value=4)
    data['ano_in'] = data['ano_in'].replace(to_replace=2014, value=5)

    train = data[data['ano_in'] < 5]
    test = data[data['ano_in'] >= 5]

    return train, test


def prepareForBoxplots(data, gender=True):
    """
    brings data into the correct format for generating a boxplot over all students from all years

    in this particular case we use year of university entrance as query_id, psu scores as features and notas as rank

    writes one dataset with protection_status "gender" and one with protection_status "highschool_type"
    """

    data = data[data['sem'] == 1]
    data = data[data['inactivo'] != 1]

    # drop all lines where values are missing
    data = data.dropna(subset=['nem', 'psu_mat', 'psu_len', 'psu_cie', 'notas_', 'uds_i_'])

    # drop all columns that are not needed
    if(gender):
        keep_cols = ['hombre', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'notas_', 'uds_i_', 'uds_r_', 'uds_e_']
    else:
        keep_cols = ['highschool_type', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'notas_', 'uds_i_', 'uds_r_', 'uds_e_']

    data = data[keep_cols]

    # replace NaNs with zeros
    data['uds_r_'].fillna(0)
    data['uds_e_'].fillna(0)

    # add new column for ranking scores
    data['score'] = np.zeros(data.shape[0])

    # calculate score based on grades and credits
    for idx, row in data.iterrows():
        grades = row.loc['notas_']
        credits_taken = row.loc['uds_i_']
        credits_failed = row.loc['uds_r_']
        credits_dropped = row.loc['uds_e_']

        score = grades * (credits_taken - credits_failed - credits_dropped) / credits_taken
        data.loc[idx, 'score'] = score

    # don't need these columns anymore
    data = data.drop(columns=['notas_', 'uds_i_', 'uds_r_', 'uds_e_'])

    # zscore psu scores and normalize scores
    data['psu_mat'] = stats.zscore(data['psu_mat'])
    data['psu_len'] = stats.zscore(data['psu_len'])
    data['psu_cie'] = stats.zscore(data['psu_cie'])
    data['nem'] = stats.zscore(data['nem'])
    data['score'] = stats.zscore(data['score'])

    # rename protected column to prot_attr
    data.columns = ['prot\_attr', 'psu\_mat', 'psu\_len', 'psu\_cie', 'nem', 'score']

    return data


