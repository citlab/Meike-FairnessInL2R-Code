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

def prepareChileDataset():
    """
    prepares Chile University dataset in a way that we can create a heatmap of it that answers the
    above mentioned research question
    """
    data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')

    # drop all uninteresting columns
    data = data.drop(columns=['rank_in', 'rank_em', 'tip_ing', 'sipee', 'bea', 'deporte', 'genero'])
    data = data[data['sem'] == 1]
    data = data[data.inactivo != 1]
    data = data.dropna(axis=1, how='any')
    return data


data = prepareChileDataset()
corr = data.corr()

f, ax = plt.subplots(figsize=(9, 6))

hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                 linewidths=.5, annot_kws={"size":8})
fig = hm.get_figure()
fig.savefig('heatmapChileMeike.png', pad_inches=1, bbox_inches='tight')
