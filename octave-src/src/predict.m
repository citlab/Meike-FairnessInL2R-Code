#!/opt/octave4.2.2/bin/octave -qf
% command line arguments:
% predict.m model feature_file

% suppress output
more off;


% load constants
addpath(".")
source "./globals.m";

%omega = load(argv(){1});
%drgfile = argv(){2};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYNTHETIC EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TOP MALE
%omega = load('../sample/synthetic/top_male_bottom_female/GAMMA=0/sample_model_gender_sep.m');
%drgfile = '../sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt';

%omega = load('../sample/synthetic/top_male_bottom_female/GAMMA=75/sample_model_gender_sep.m');
%drgfile = '../sample/synthetic/top_male_bottom_female/GAMMA=75/sample_test_data_scoreAndGender_separated.txt';

%omega = load('../sample/synthetic/top_male_bottom_female/GAMMA=150/sample_model_gender_sep.m');
%drgfile = '../sample/synthetic/top_male_bottom_female/GAMMA=150/sample_test_data_scoreAndGender_separated.txt';

% TOP FEMALE
%omega = load('../sample/synthetic_score_gender/top_female_bottom_male/sample_model_gender_sep.m');
%drgfile = '../sample/synthetic_score_gender/top_female_bottom_male/sample_test_data_scoreAndGender_separated.txt';
%omega = load('../sample/synthetic_score_gender/distribution_based/sample_model_gender_normdist.m');
%drgfile = '../sample/synthetic_score_gender/distribution_based/sample_test_data_scoreAndGender_normalDistribution.txt';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHILE EXPERIMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GENDER
%omega = load('../sample/ChileUni/GAMMA=0/chileDataL2R_gender_model.m');
%drgfile = '../sample/ChileUni/GAMMA=0/chileDataL2R_gender_test.txt';

%omega = load('../sample/ChileUni/GAMMA=100000/chileDataL2R_gender_model.m');
%drgfile = '../sample/ChileUni/GAMMA=100000/chileDataL2R_gender_test.txt';

%omega = load('../sample/ChileUni/GAMMA=5000000/chileDataL2R_gender_model.m');
%drgfile = '../sample/ChileUni/GAMMA=5000000/chileDataL2R_gender_test.txt';

% HIGHSCHOOL
%omega = load('../sample/ChileUni/GAMMA=0/chileDataL2R_highschool_model.m');
%drgfile = '../sample/ChileUni/GAMMA=0/chileDataL2R_highschool_test.txt';

%omega = load('../sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_model.m');
%drgfile = '../sample/ChileUni/GAMMA=100000/chileDataL2R_highschool_test.txt';

%omega = load('../sample/ChileUni/GAMMA=5000000/chileDataL2R_highschool_model.m');
%drgfile = '../sample/ChileUni/GAMMA=5000000/chileDataL2R_highschool_test.txt';

% colorblind done in Cao original code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TREC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drgfile = "../sample/TREC/GAMMA=0/features_with_total_order-zscore-test.csv";
omega = load("../sample/TREC/GAMMA=0/features_with_total_order-zscore_model.m");

%drgfile = "../sample/TREC/GAMMA=10000/features_with_total_order-zscore-test.csv";
%omega = load("../sample/TREC/GAMMA=10000/features_with_total_order-zscore_model.m");

%drgfile = "../sample/TREC/GAMMA=500000/features_with_total_order-zscore-test.csv";
%omega = load("../sample/TREC/GAMMA=500000/features_with_total_order-zscore_model.m");

drg = load(drgfile);

list_id = drg(:,1);
X = drg(:,2:size(drg,2)-1);


z =  X * omega.omega;
doc_ids = 1:size(z);

# also write y for later evaluation
y = drg(:, size(drg,2));
y = [list_id, doc_ids', y, X(:, PROT_COL)];

filename = [drgfile "_ORIG.pred"];
dlmwrite(filename, y);


# add protection status to a for later evaluation
z = [z, X(:, PROT_COL)];

# add list ids and document ids for later evaluation
z = [list_id, doc_ids', z];

unsorted_ranks = z;
filename = [drgfile "_UNSORTED.pred"];
dlmwrite(filename, unsorted_ranks);

# add a little random to avoid ties
r = @(i) (i+rand*0.02-0.01);

for id = unique(list_id)'
    indexes = find(list_id==id);
    z_temp = z(indexes, :);
    z(indexes, :) = sortrows(z_temp, -3);
endfor
sorted_ranks = z;
filename = [drgfile "_SORTED.pred"];

dlmwrite(filename, sorted_ranks)
figure(); plot(z);