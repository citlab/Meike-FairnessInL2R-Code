#!/opt/octave4.2.2/bin/octave -qf
% train a linear network using the examples provided in the training_set file
% and writes the model on output_model
%
% usage: # train.m training_set output_model

% suppress output
more off;

% pararrayfun is in the 'general' package
pkg load general;
pkg load parallel;

% read arguments on the command line
arg_list = argv ();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYNTHETIC EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training_file = '../sample/synthetic_score_gender/distribution_based/sample_train_data_scoreAndGender_normalDistribution.txt'
%model_file = '../sample/synthetic_score_gender/distribution_based/sample_model_gender_normdist.m'
%training_file = '../sample/synthetic_score_gender/top_male_bottom_female/sample_train_data_scoreAndGender_separated.txt'
%model_file = '../sample/synthetic_score_gender/top_male_bottom_female/sample_model_gender_sep.m'
%training_file = '../sample/synthetic_score_gender/top_female_bottom_male/sample_train_data_scoreAndGender_separated.txt'
%model_file = '../sample/synthetic_score_gender/top_female_bottom_male/sample_model_gender_sep.m'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHILE EXPERIMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training_file = '../sample/ChileUni/chileDataL2R_gender_train.txt'
%model_file = '../sample/ChileUni/chileDataL2R_gender_model.m'
%training_file = '../sample/ChileUni/chileDataL2R_highschool_train.txt'
%model_file = '../sample/ChileUni/chileDataL2R_highschool_model.m'

% HIGHSCHOOL
training_file = '../sample/ChileUni/GAMMA=0/chileDataL2R_highschool_train.txt'
model_file = '../sample/ChileUni/GAMMA=0/chileDataL2R_highschool_model.m'

% COLORBLIND
%training_file = '../sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_train.txt'
%model_file = '../sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_model.m'


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TREC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training_file = "../sample/TREC/GAMMA=0/features_with_total_order-zscore-train.csv";
%model_file = "../sample/TREC/GAMMA=0/features_with_total_order-zscore_model.m";

%training_file = "../sample/TREC/GAMMA=10000/features_with_total_order-zscore-train.csv";
%model_file = "../sample/TREC/GAMMA=10000/features_with_total_order-zscore_model.m";

%training_file = "../sample/TREC/GAMMA=500000/features_with_total_order-zscore-train.csv";
%model_file = "../sample/TREC/GAMMA=500000/features_with_total_order-zscore_model.m";


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TOY DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training_file = '../sample/toy_data/toy_training_data.m'
%model_file = '../sample/toy_data/toy_model.m'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CAO'S SAMPLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training_file = '../sample/sample_training_data.m'
%model_file = '../sample/sample_model.m'
%training_file = arg_list{1,1};
%model_file = arg_list{2,1};

% load constants
addpath(".")
source "./globals.m";

% load training dataset
disp('loading training data...')
data = load(training_file);
list_id = data(:,1);
X = data(:,2:size(data,2)-1);
y = data(:,size(data,2));

% launch the training routine
disp(sprintf('training, %d iteration, %d examples, learning rate %f...', T, size(X,1), e))
tic();
omega = trainNN(list_id, X, y, T, e);
training_time = toc();
disp(sprintf('finished training, time elapsed: %d seconds', training_time))
save(model_file, "omega");
