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
k = 100

input_file1 = '../../octave-src/sample/synthetic_score_gender/top_female_bottom_male/sample_train_data_scoreAndGender_separated.txt'
input_file2 = '../../octave-src/sample/synthetic_score_gender/top_female_bottom_male/sample_test_data_scoreAndGender_separated_GAMMA_ZERO.txt.pred'
input_file3 = '../../octave-src/sample/synthetic_score_gender/top_female_bottom_male/sample_test_data_scoreAndGender_separated_GAMMA_MEDIUM.txt.pred'
input_file4 = '../../octave-src/sample/synthetic_score_gender/top_female_bottom_male/sample_test_data_scoreAndGender_separated_GAMMA_LARGE.txt.pred'
output_file = '../../plots/synthetic/separated/top_female_bottom_male/rankings_group_property_only.png'


mpl.rcParams.update({'font.size': 12, 'lines.linewidth': 3, 'lines.markersize': 15, 'font.family':'Times New Roman'})
# avoid type 3 (i.e. bitmap) fonts in figures
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True


f, ax = plt.subplots()

# Generate dummy info for plot handles "h"
red_circle, = plt.plot(1, 0, 'ro', ms=2, label="protected")
blue_plus, = plt.plot(1, 0, 'b+', ms=4, label="non-protected");

data = pd.read_csv(input_file1, sep=",", header=None)
for i in range(0, k, 2):
    if data.iloc[i][1] == PROT_ATTR:
        plt.plot(1, i, 'ro', ms=2);  # , 'markersize', 5, 'markerfacecolor', 'r');
    else:
        plt.plot(1, i, 'b+', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

data = pd.read_csv(input_file2, sep=",", header=None)
for i in range(0, k, 2):
    if data.iloc[i][PROT_COL] == PROT_ATTR:
        plt.plot(2, i, 'ro', ms=2);  # , 'markersize', 5, 'markerfacecolor', 'r');
    else:
        plt.plot(2, i, 'b+', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

data = pd.read_csv(input_file3, sep=",", header=None)
for i in range(0, k, 2):
    if data.iloc[i][PROT_COL] == PROT_ATTR:
        plt.plot(3, i, 'ro', ms=2);  # , 'markersize', 5, 'markerfacecolor', 'r');
    else:
        plt.plot(3, i, 'b+', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

data = pd.read_csv(input_file4, sep=",", header=None)
for i in range(0, k, 2):
    if data.iloc[i][PROT_COL] == PROT_ATTR:
        plt.plot(4, i, 'ro', ms=2);  # , 'markersize', 5, 'markerfacecolor', 'r');
    else:
        plt.plot(4, i, 'b+', ms=4);  # , 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

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

