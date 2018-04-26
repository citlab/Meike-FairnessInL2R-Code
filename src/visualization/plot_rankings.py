'''
Created on Apr 26, 2018

@author: meike.zehlike
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROT_COL = 0
PROT_ATTR = 1
k = 100

input_file1 = '../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_train_data_scoreAndGender_separated.txt'
input_file2 = '../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_test_data_scoreAndGender_separated_GAMMA_ZERO.txt.pred'
input_file3 = '../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_test_data_scoreAndGender_separated_GAMMA_MEDIUM.txt.pred'
input_file4 = '../../octave-src/sample/synthetic_score_gender/top_male_bottom_female/sample_test_data_scoreAndGender_separated_GAMMA_LARGE.txt.pred'
output_file = '../../octave-src/plots/synthetic/separated/rankings_group_property_only.png'


f, ax = plt.subplots()
# #   output_file_with_extension = strcat(output_file, ".png");
#
#   # Generate dummy info for plot handles "h"
# h = np.zeros(2,1);
# h(1,1) = plt.plot(1,1,'ro', 'markersize', 5, 'markerfacecolor', 'r');hold on;
# h(2,1) = plt.plot(1,1,'b+', 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);

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

plt.xticks(np.arange(1, 5, step=1), ('Training Data', 'Normal L2R', 'Small Lambda', 'Large Lambda'))
# legend(h, 'protected', 'non-protected');
#
# title (title1);
# print(hf, output_file_with_extension);
# %print (hf, output_file_with_extension, "-dpng");
# %system (sprintf("pdflatex %s", output_file));
# open(output_file_with_extension);

plt.savefig(output_file)
plt.show()

