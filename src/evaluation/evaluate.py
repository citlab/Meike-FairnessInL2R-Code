'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from nose import result

PROT_ATTR = 1

##################################################################################
# EVALUATE PRECENTAGE OF PROTECTED CANDIDATES PER CHUNK
##################################################################################

##################################################################################
# EVALUATE EXPOSURE DIFFERENCE BETWEEN GROUPS
##################################################################################

##################################################################################
# EVALUATE KENDALL'S TAU
##################################################################################
def evaluate(prediction, original, result_filename):
    predictedGroups = prediction.groupby(prediction['query_id'], as_index=False, sort=False)
    originalGroups = original.groupby(prediction['query_id'], as_index=False, sort=False)

    result = pd.DataFrame(np.nan,
                          index=range(0, len(predictedGroups)),
                          columns=['query_id', 'exposure_prot_pred', 'exposure_nprot_pred', 'exp_diff_pred',
                                   'exposure_prot_orig', 'exposure_nprot_orig', 'exp_diff_orig',
                                   'kendall_tau', 'p_value'])

    i = 0
    for name, predGroup in predictedGroups:
        origGroup = originalGroups.get_group(name)
        x = result.iloc[i]
        result.loc[i]['query_id'] = name
        result.loc[i]['exposure_prot_pred'] = calculate_group_exposure(predGroup, origGroup)[0]
        result.loc[i]['exposure_nprot_pred'] = calculate_group_exposure(predGroup, origGroup)[1]
        result.loc[i]['exp_diff_pred'] = calculate_group_exposure(predGroup, origGroup)[2]
        result.loc[i]['exposure_prot_orig'] = calculate_group_exposure(predGroup, origGroup)[3]
        result.loc[i]['exposure_nprot_orig'] = calculate_group_exposure(predGroup, origGroup)[4]
        result.loc[i]['exp_diff_orig'] = calculate_group_exposure(predGroup, origGroup)[5]
        result.loc[i]['kendall_tau'] = stats.kendalltau(origGroup['doc_id'], predGroup['doc_id'])[0]
        result.loc[i]['p_value'] = stats.kendalltau(origGroup['doc_id'], predGroup['doc_id'])[1]
        i += 1

    with open(result_filename, "w") as text_file:
        print(result, file=text_file)
    return


def calculate_group_exposure(prediction, original):
    prediction = pd.DataFrame(index=range(4), columns=['prot_attr', 'prediction'])
    prediction['prot_attr'] = [0, 0, 1, 1]
    prediction['prediction'] = [0.1, 0.2, 0.3, 0.4]

    # exposure in predictions
    prot_rows_pred = prediction.loc[prediction['prot_attr'] == PROT_ATTR]
    nprot_rows_pred = prediction.loc[prediction['prot_attr'] != PROT_ATTR]

    prot_exp_per_doc = np.exp(prot_rows_pred['prediction']) / sum(np.exp(prediction['prediction']))
    avg_prot_exp_pred = sum(prot_exp_per_doc) / prediction.shape[0]

    nprot_exp_per_doc = np.exp(nprot_rows_pred['prediction']) / sum(np.exp(prediction['prediction']))
    avg_nprot_exp_pred = sum(nprot_exp_per_doc) / prediction.shape[0]

    # exposure in original
    prot_rows_orig = original.loc[original['prot_attr'] == PROT_ATTR]
    nprot_rows_orig = original.loc[original['prot_attr'] != PROT_ATTR]

    prot_exp_per_doc = 1 - np.exp(prot_rows_orig['prediction']) / sum(np.exp(original['prediction']))
    avg_prot_exp_orig = sum(prot_exp_per_doc) / original.shape[0]

    nprot_exp_per_doc = 1 - np.exp(nprot_rows_orig['prediction']) / sum(np.exp(original['prediction']))
    avg_nprot_exp_orig = sum(nprot_exp_per_doc) / original.shape[0]

    return avg_prot_exp_pred, avg_nprot_exp_pred, (avg_nprot_exp_pred - avg_prot_exp_pred), avg_prot_exp_orig, avg_nprot_exp_orig, (avg_nprot_exp_orig - avg_prot_exp_orig)



def protected_percentage_per_chunk(ranking, chunksize, plot_filename):
    '''
    calculates percentage of protected (non-protected resp.) for each chunk of the ranking
    plots them into a figure
    '''
    rankingsPerQuery = ranking.groupby(ranking['query_id'], as_index=False, sort=False)
    for name, rank in rankingsPerQuery:

        filename = plot_filename[:-4] + "_" + str(name) + plot_filename[-4:]

        rowNum = rank.shape[0]
        chunkStartPositions = np.arange(0, rowNum, chunksize)

        result_protected = np.empty(len(chunkStartPositions))
        result_nonprotected = np.empty(len(chunkStartPositions))

        total_protected = rank['prot_attr'].value_counts()[1]
        total_nonprotected = rank['prot_attr'].value_counts()[0]

        for idx, start in enumerate(chunkStartPositions):
            if idx == (len(chunkStartPositions) - 1):
                # last Chunk
                end = rowNum
            else:
                end = chunkStartPositions[idx + 1]
            chunk = rank.iloc[start:end]

            try:
                chunk_protected = chunk['prot_attr'].value_counts()[1]
            except KeyError:
                # no protected elements in this chunk
                chunk_protected = 0

            try:
                chunk_nonprotected = chunk['prot_attr'].value_counts()[0]
            except KeyError:
                # no nonprotected elements in this chunk
                chunk_nonprotected = 0

            result_protected[idx] = chunk_protected / rowNum
            result_nonprotected[idx] = chunk_nonprotected / rowNum


        plot_protected_percentage_per_chunk(result_protected, result_nonprotected, len(chunkStartPositions), chunkStartPositions, filename, chunksize / 2)
    return




def plot_protected_percentage_per_chunk(prot, nonprot, tick_length, x_ticks, plot_filename, bar_width):
    mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True


    f, ax = plt.subplots(figsize=(20, 10))
#     plt.plot(prot, 'r-')
#     plt.plot(nonprot, 'b')
    width = bar_width

    ax.bar(x_ticks, prot, color='r', width=width, hatch='//', edgecolor='white')
    ax.bar(x_ticks + width, nonprot, color='b', width=width)

    # plt.xticks(np.arange(tick_length))
    # ax.set_xticklabels(x_ticks)

    plt.xlabel ("ranking range");
    plt.ylabel("ratio per chunk")
    plt.legend(['protected', 'non-protected'])
#     plt.show()

    plt.savefig(plot_filename, bbox_inches='tight')




