'''
Created on Apr 11, 2018

@author: meike.zehlike

creates a heatmap from the Chile University dataset to show correlations between highschool grades
as well as university entrance tests and the success in university after a year

also asks if the success after one year is gender specific or schooling specific (public vs semi-
public vs private schools)
'''

import matplotlib.pyplot as plt
import seaborn as sns


def cool_warm_heatmap(data, filename):
    corr = data.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
        linewidths=.5, annot_kws={"size":8})
    fig = hm.get_figure()
    fig.savefig(filename, pad_inches=1, bbox_inches='tight')
