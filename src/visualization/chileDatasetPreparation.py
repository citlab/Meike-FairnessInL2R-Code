'''
Created on Apr 17, 2018

@author: mzehlike
'''
import numpy as np


def principalDataPreparation(data):
    # drop all uninteresting columns
    data = data.drop(columns=['rank_in', 'rank_em', 'tip_ing', 'sipee', 'bea', 'deporte', 'genero'])
    data = data.fillna(value={'uds_e_':0})  # fill NaNs with 0 in this specific column only

    # merge muni, sub and part column into one highschool type column
    data['muni'] = data['muni'].replace([0], np.nan)
    data['muni'] = data['muni'].replace([1], 0)
    data['sub'] = data['sub'].replace([0], np.nan)
    data['part'] = data['part'].replace([0], np.nan)
    data['part'] = data['part'].replace([1], 2)
    data['muni'] = data['muni'].fillna(data['part'])
    data['muni'] = data['muni'].fillna(data['sub'])
    data = data.rename(index=str, columns={"muni":"highschool_type"})
    data = data.drop(columns=['sub', 'part'])

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
