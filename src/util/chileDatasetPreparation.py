'''
Created on Apr 17, 2018

@author: mzehlike
'''
import numpy as np
from conda._vendor.auxlib.collection import first


def principalDataPreparation(data):
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

def prepareForL2R(data, gender=True):
    """
    brings data into the correct format for L2R octave code with following scheme
    query_id; protection_status; feature_1; ...; feature_n; rank

    in this particular case we use year of university entrance as query_id, psu scores as features and notas as rank

    writes one dataset with protection_status "gender" and one with protection_status "highschool_type"
    """

    def rank(x, sortby):
        x.sort_values([sortby[0], sortby[1]], ascending=[False, False], inplace=True)
        x['rank'] = range(x.shape[0])
        # normalize ranks
        x['rank'] = x['rank'] / x.shape[0]
        return x

    def normalize_values(x):
        return x / x.max()
        return x

    data = data[data['sem'] == 1]
    data = data[data['inactivo'] != 1]

    # drop all lines where values are missing
    data = data.dropna(subset=['nem', 'psu_mat', 'psu_len', 'psu_cie', 'notas_', 'uds_i_'])

    # drop all columns that are not needed
    if(gender):
        keep_cols = ['ano_in', 'hombre', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'notas_', 'uds_i_']
    else:
        keep_cols = ['ano_in', 'highschool_type', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'notas_', 'uds_i_']

    data = data[keep_cols]

    # add new column for ranks
    data['rank'] = np.zeros(data.shape[0])

    # group years together and rank them by notas
    sortby = ['notas_', 'uds_i_']  # only used because not possible somehow to have more than one parameter
    data = data.groupby(data['ano_in'], as_index=False, sort=False).apply(rank, (sortby))

    # drop notas_ and uds_i_ because not needed anymore but would be seen as features
    data = data.drop(columns=sortby)

    # normalize scores (ranks already normalized)
    data[['psu_mat', 'psu_len', 'psu_cie', 'nem']] = data.groupby(data['ano_in'], as_index=False, sort=False)[['psu_mat', 'psu_len', 'psu_cie', 'nem']].transform(lambda x: x / x.max())

    # replace ano_in with query_ids that start from 1
    data['ano_in'] = data['ano_in'].replace(to_replace=2010, value=1)
    data['ano_in'] = data['ano_in'].replace(to_replace=2011, value=2)
    data['ano_in'] = data['ano_in'].replace(to_replace=2012, value=3)
    data['ano_in'] = data['ano_in'].replace(to_replace=2013, value=4)
    data['ano_in'] = data['ano_in'].replace(to_replace=2014, value=5)

    return data


