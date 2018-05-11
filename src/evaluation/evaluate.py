'''
Created on May 11, 2018

@author: mzehlike
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

##################################################################################
# EVALUATE PRECENTAGE OF PROTECTED CANDIDATES PER CHUNK
##################################################################################

##################################################################################
# EVALUATE EXPOSURE DIFFERENCE BETWEEN GROUPS
##################################################################################

##################################################################################
# EVALUATE KENDALL'S TAU
##################################################################################

def calculate_kendalls_tau(prediction, original, filename):
    predictedGroups = prediction.groupby(prediction['query_id'], as_index=False, sort=False)
    originalGroups = original.groupby(prediction['query_id'], as_index=False, sort=False)

    tau_dict = {}

    for name, predGroup in predictedGroups:
        origGroup = originalGroups.get_group(name)
        tau_dict[name] = stats.kendalltau(origGroup, predGroup)

    with open(filename, "w") as text_file:
        print(tau_dict, file=text_file)
    return


def exposureDiff(ranking):

    return 0


def calculate_protected_percentage_per_chunk(ranking, chunksize, plot_filename):
    '''
    calculates percentage of protected (non-protected resp.) for each chunk of the ranking
    writes them to txt file
    plots them into a figure
    returns them in two separate numpy arrays
    '''
    rowNum = ranking.shape[0]
    chunkStartPositions = np.arange(0, rowNum, chunksize)

    result_protected = np.empty(len(chunkStartPositions))
    result_nonprotected = np.empty(len(chunkStartPositions))

    total_protected = ranking['prot_attr'].value_counts()[1]
    total_nonprotected = ranking['prot_attr'].value_counts()[0]

    for idx, start in enumerate(chunkStartPositions):
        if idx == (len(chunkStartPositions) - 1):
            # last Chunk
            end = rowNum
        else:
            end = chunkStartPositions[idx + 1]
        chunk = ranking.iloc[start:end]

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

        result_protected[idx] = chunk_protected / total_protected
        result_nonprotected[idx] = chunk_nonprotected / total_nonprotected


    plot_protected_percentage_per_chunk(result_protected, result_nonprotected, len(chunkStartPositions), chunkStartPositions, plot_filename)
    return




def plot_protected_percentage_per_chunk(prot, nonprot, tick_length, x_ticks, plot_filename):
    mpl.rcParams.update({'font.size': 30, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True


    f, ax = plt.subplots(figsize=(20, 10))
#     plt.plot(prot, 'r-')
#     plt.plot(nonprot, 'b')
    width = 10

    ax.bar(x_ticks, prot, color='r', width=width, hatch='//', edgecolor='white')
    ax.bar(x_ticks + width, nonprot, color='b', width=width)

    # plt.xticks(np.arange(tick_length))
    # ax.set_xticklabels(x_ticks)

    plt.xlabel ("ranking range");
    plt.ylabel("ratio per chunk")
    plt.legend(['protected', 'non-protected'])
    plt.show()

    plt.savefig(plot_filename, bbox_inches='tight')




