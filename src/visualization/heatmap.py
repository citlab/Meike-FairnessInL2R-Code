'''
Created on Apr 11, 2018

@author: meike.zehlike

creates a heatmap from the Chile University dataset to show correlations between highschool grades
as well as university entrance tests and the success in university after a year

also asks if the success after one year is gender specific or schooling specific (public vs semi-
public vs private schools)
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def successfulStudents():
    """
    this heatmap considers only students that have succeeded in the first semester of university
    success means that they took at least 45 credits and failed (or dropped) at most 10 of them
    """
    data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')

    # drop all uninteresting columns
    data = data.drop(columns=['rank_in', 'rank_em', 'tip_ing', 'sipee', 'bea', 'deporte', 'genero', 'rat_ud'])
    data = data.fillna(value={'uds_e_':0})  # fill NaNs with 0 in this specific column only
    data = data[data['sem'] == 1]
    data = data[data['inactivo'] != 1]
    data = data[data['uds_e_'] <= 10]
    # after the above steps these columns contain all the same value and hence can be dropped
    data = data.drop(columns=['sem', 'inactivo', 'uds_e_'])
    data = data.dropna(subset=['nem', 'psu_mat', 'psu_len', 'psu_cie', 'psu_pond', 'notas_'])
    data = data[data['uds_i_'] >= 45]
    data = data[data['uds_r_'] <= 10]

    corr = data.corr()
    f, ax = plt.subplots(figsize=(9, 6))

    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                     linewidths=.5, annot_kws={"size":8})
    fig = hm.get_figure()
    fig.savefig('heatmapSuccessfulStudents.png', pad_inches=1, bbox_inches='tight')


def allStudents():
    """
    this heatmap considers all students that have taken exams after the first semester of university
    this means they were not "inactive" and non of their values in the relevant columns were NaN
    """
    data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')

    # drop all uninteresting columns
    data = data.drop(columns=['rank_in', 'rank_em', 'tip_ing', 'sipee', 'bea', 'deporte', 'genero', 'rat_ud'])
    data = data.fillna(value={'uds_e_':0})  # fill NaNs with 0 in this specific column only
    data = data[data['sem'] == 1]
    data = data[data['inactivo'] != 1]
        # after the above steps these columns contain all the same value and hence can be dropped
    data = data.drop(columns=['sem', 'inactivo'])
    data = data.dropna(subset=['nem', 'psu_mat', 'psu_len', 'psu_cie', 'psu_pond', 'notas_', 'uds_i_', 'uds_r_', 'uds_e_'])

    corr = data.corr()
    f, ax = plt.subplots(figsize=(9, 6))

    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                     linewidths=.5, annot_kws={"size":8})
    fig = hm.get_figure()
    fig.savefig('heatmapAllStudents.png', pad_inches=1, bbox_inches='tight')


successfulStudents()
allStudents()

