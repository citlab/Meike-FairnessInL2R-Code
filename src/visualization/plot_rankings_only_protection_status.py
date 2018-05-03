'''
Created on Apr 26, 2018

@author: meike.zehlike
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PROT_COL = 0
PROT_ATTR = 1


def plot_rankings(input_file1, input_file2, input_file3, input_file4, output_file, k, step_size):
    mpl.rcParams.update({'font.size': 12, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True


    f, ax = plt.subplots()

    # Generate dummy info for plot handles "h"
    red_circle, = plt.plot(1, 0, 'ro', ms=1, label="protected")
    blue_plus, = plt.plot(1, 0, 'b+', ms=1, label="non-protected");

    data = pd.read_csv(input_file1, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][1] == PROT_ATTR:
            plt.plot(1, i, 'ro', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'r');
        else:
            plt.plot(1, i, 'b+', ms=6);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

    data = pd.read_csv(input_file2, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][PROT_COL] == PROT_ATTR:
            plt.plot(2, i, 'ro', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'r');
        else:
            plt.plot(2, i, 'b+', ms=6);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

    data = pd.read_csv(input_file3, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][PROT_COL] == PROT_ATTR:
            plt.plot(3, i, 'ro', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'r');
        else:
            plt.plot(3, i, 'b+', ms=6);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

    data = pd.read_csv(input_file4, sep=",", header=None)
    for i in range(0, k, step_size):
        if data.iloc[i][PROT_COL] == PROT_ATTR:
            plt.plot(4, i, 'ro', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'r');
        else:
            plt.plot(4, i, 'b+', ms=6);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

    plt.gca().invert_yaxis()
    plt.ylabel ("ranking position");
    plt.title("Predicted Positions of L2R Without and With Fairness Constraint")
    plt.xticks(np.arange(1, 5, step=1), ('Training Data', 'Normal L2R', 'Small Gamma', 'Large Gamma'))
    plt.legend([red_circle, blue_plus], ['protected', 'non-protected'])

    # legend(h, 'protected', 'non-protected');
    #
    # title (title1);
    # print(hf, output_file_with_extension);
    # %print (hf, output_file_with_extension, "-dpng");
    # %system (sprintf("pdflatex %s", output_file));
    # open(output_file_with_extension);

    plt.savefig(output_file)
    plt.show()

