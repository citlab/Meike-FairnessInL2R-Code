'''
Created on Apr 17, 2018

@author: mzehlike
'''

import util.chileDatasetPreparation as prep
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

##############################################################################################
# BOXPLOTS
##############################################################################################

# write dataset for gender
data = pd.read_excel('../../data/ChileUniversity/UCH-FCFM-GRADES_2010_2014_chato.xls.xlsx')
data = prep.principalDataPreparation(data)
data = prep.prepareForBoxplots(data)

def boxPlot(data, filename):
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

    plt.xlabel ("position");
    plt.ylabel("proportion")
    plt.legend(['protected', 'non-protected'])
#     plt.show()

    plt.savefig(plot_filename, bbox_inches='tight')










