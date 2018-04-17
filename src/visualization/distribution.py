'''
Created on Apr 17, 2018

@author: mzehlike
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools


def determineGroups(attributeNamesAndCategories):
    elementSets = []
    groups = []
    for attr, cardinality in attributeNamesAndCategories.items():
        elementSets.append(list(range(0, cardinality)))

    groups = list(itertools.product(*elementSets))
    return groups


def separate_groups(data_set, categories, attributeItems):
    num_categories = len(categories)
    separateByGroups = [[] for _ in range(num_categories)]

    for idx, row in data_set.iterrows():
        categorieList = []
        for j in attributeItems:
            col_name = j[0]
            categorieList.append(row[col_name])
        separateByGroups[categories.index(tuple(categorieList))].append(row)
        categorieList = []
    return separateByGroups


def plot(data_set, attributeNamesAndCategories, attributeQuality, filename, labels):
    colors = ['green', 'red', 'blue']
    markers = ['-o', '-<', '-s', '-+', '-d']
#     label=['Male-married','Male-single','Male-divorced','Female']
#     label=['German','Turkish','Yugoslavian','Greek','Italian']
#     label=['German','Other','Asylum','EU Country']
#     label=['Male single','Female divorced/separated/married','Male divorced/separated','Male married/widowed']
    data = data_set.sort_values(by=attributeQuality, ascending=False)
    best = data[attributeQuality].iloc[0]
    categories = determineGroups(attributeNamesAndCategories)
    attributeItems = attributeNamesAndCategories.items()
    output_ranking_separated = separate_groups(data, categories, attributeItems)
    separateQualityByGroups = []
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(211)
    round_2f = []
    for idx, row in data.iterrows():
        row[attributeQuality] = float(row[attributeQuality]) / best

    for i in range(len(categories)):
        separateQualityByGroups.append([quality[attributeQuality] for quality in output_ranking_separated[i]])
        fit = stats.norm.pdf(separateQualityByGroups[i], np.mean(separateQualityByGroups[i]), np.std(separateQualityByGroups[i]))
#        plt.plot(separateQualityByGroups[i], fit, markers[i], markersize=6, label=categories[i], color=colors[i])
        plt.plot(separateQualityByGroups[i], fit, markers[i], markersize=6, label=labels[i], color=colors[i])
        plt.legend(loc='center left', fontsize='x-large', bbox_to_anchor=(1, 0.5))
        round_2f.append([round(elem, 2) for elem in separateQualityByGroups[i]])
    plt.xlabel(attributeQuality + ' (Quality)')
    plt.ylabel('Probability Density Function')

    plt.subplot(212)
#    plt.hist(round_2f, 30, histtype='bar', label=categories, color=colors[:len(categories)])
    plt.hist(round_2f, 30, histtype='bar', label=labels, color=colors[:len(categories)])
    plt.xlabel(attributeQuality + ' (Quality)')
    plt.ylabel('Frequency')
    plt.legend(loc='center left', fontsize='x-large', bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, dpi=100)



