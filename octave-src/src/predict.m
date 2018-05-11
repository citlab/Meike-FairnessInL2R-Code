#!/opt/octave4.2.2/bin/octave -qf
% command line arguments:
% predict.m model feature_file

% suppress output
more off;


% load constants
addpath(".")
source "./globals.m";

global GAMMA;
%omega = load(argv(){1});
%drgfile = argv(){2};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYNTHETIC EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%omega = load('../sample/synthetic_score_gender/top_female_bottom_male/sample_model_gender_sep.m');
%drgfile = '../sample/synthetic_score_gender/top_female_bottom_male/sample_test_data_scoreAndGender_separated.txt';
%omega = load('../sample/synthetic_score_gender/top_female_bottom_male/sample_model_gender_sep.m');
%drgfile = '../sample/synthetic_score_gender/top_female_bottom_male/sample_test_data_scoreAndGender_separated.txt';
%omega = load('../sample/synthetic_score_gender/distribution_based/sample_model_gender_normdist.m');
%drgfile = '../sample/synthetic_score_gender/distribution_based/sample_test_data_scoreAndGender_normalDistribution.txt';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHILE EXPERIMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega = load('../sample/ChileUni/chileDataL2R_gender_model_GAMMA0.m');
drgfile = '../sample/ChileUni/chileDataL2R_gender_test.txt';

%omega = load('../sample/ChileUni/chileDataL2R_highschool_model_GAMMA1000.m');
%drgfile = '../sample/ChileUni/chileDataL2R_highschool_test.txt';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TREC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drgfile = "../sample/TREC/GAMMA=0/features_with_total_order-zscore-v2_subset_test.csv";
omega = load("../sample/TREC/GAMMA=0/features_with_total_order-zscore-v2_model_subset.m");

drg = load(drgfile);

list_id = drg(:,1);
X = drg(:,2:size(drg,2)-1);

z =  X * omega.omega;
# add protection status to a for later evaluation
z = [z, X(:, PROT_COL)];

# add a little random to avoid ties
r = @(i) (i+rand*0.02-0.01);

for id = unique(list_id)'
    indexes = find(list_id==id);
    z_temp = z(indexes, :);
    z(indexes, :) = sortrows(z_temp, 1);
endfor
ranks = z;
gamma_char = sprintf("%d", GAMMA);
filename = [drgfile ".GAMMA" sprintf("%d", GAMMA) ".pred"];

dlmwrite(filename, ranks)
figure(); plot(z);