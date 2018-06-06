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
arg_list = argv()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYNTHETIC EXPERIMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TOP MALE BOTTOM FEMALE
%directory = "../sample/synthetic/top_male_bottom_female/GAMMA=0/";
%training_file = [directory 'sample_train_data_scoreAndGender_separated.txt']
%model_file = [directory 'sample_model_gender_sep.m']

%directory = "../sample/synthetic/top_male_bottom_female/GAMMA=75/";
%training_file = [directory 'sample_train_data_scoreAndGender_separated.txt']
%model_file = [directory 'sample_model_gender_sep.m']

%directory = "../sample/synthetic/top_male_bottom_female/GAMMA=150/";
%training_file = [directory 'sample_train_data_scoreAndGender_separated.txt']
%model_file = [directory 'sample_model_gender_sep.m']

% TOP FEMALE BOTTOM MALE
%directory = "../sample/synthetic_score_gender/top_female_bottom_male/";
%training_file = [directory 'sample_train_data_scoreAndGender_separated.txt']
%model_file = [directory 'sample_model_gender_sep.m']

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHILE EXPERIMENT WITH SEMI_PRIVATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GENDER
%directory = "../sample/ChileUni/Semi/GAMMA=0/";
%training_file = [directory 'chileDataL2R_gender_semi_train.txt']
%model_file = [directory 'chileDataL2R_gender_semi_model.m']

%directory = "../sample/ChileUni/Semi/GAMMA=100000/";
%training_file = [directory 'chileDataL2R_gender_semi_train.txt']
%model_file = [directory 'chileDataL2R_gender_semi_model.m']

%directory = "../sample/ChileUni/Semi/GAMMA=5000000/";
%training_file = [directory 'chileDataL2R_gender_semi_train.txt']
%model_file = [directory 'chileDataL2R_gender_semi_model.m']

% HIGHSCHOOL
%directory = "../sample/ChileUni/NoSemi/GAMMA=0/";
%training_file = [directory 'chileDataL2R_highschool_semi_train.txt']
%model_file = [directory 'chileDataL2R_highschool_semi_model.m']

%directory = "../sample/ChileUni/NoSemi/GAMMA=100000/";
%training_file = [directory 'chileDataL2R_highschool_semi_train.txt']
%model_file = [directory 'chileDataL2R_highschool_semi_model.m']

%directory = "../sample/ChileUni/NoSemi/GAMMA=5000000/";
%training_file = [directory 'chileDataL2R_highschool_semi_train.txt']
%model_file = [directory 'chileDataL2R_highschool_semi_model.m']


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHILE EXPERIMENT WITHOUT SEMI_PRIVATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GENDER
%directory = "../sample/ChileUni/NoSemi/GAMMA=0/";
%training_file = [directory 'chileDataL2R_gender_nosemi_train.txt']
%model_file = [directory 'chileDataL2R_gender_nosemi_model.m']

%directory = "../sample/ChileUni/NoSemi/GAMMA=100000/";
%training_file = [directory 'chileDataL2R_gender_nosemi_train.txt']
%model_file = [directory 'chileDataL2R_gender_nosemi_model.m']

%directory = "../sample/ChileUni/NoSemi/GAMMA=5000000/";
%training_file = [directory 'chileDataL2R_gender_nosemi_train.txt']
%model_file = [directory 'chileDataL2R_gender_nosemi_model.m']

% HIGHSCHOOL
%directory = "../sample/ChileUni/NoSemi/GAMMA=0/";
%training_file = [directory 'chileDataL2R_highschool_nosemi_train.txt']
%model_file = [directory 'chileDataL2R_highschool_nosemi_model.m']

%directory = "../sample/ChileUni/NoSemi/GAMMA=100000/";
%training_file = [directory 'chileDataL2R_highschool_nosemi_train.txt']
%model_file = [directory 'chileDataL2R_highschool_nosemi_model.m']

%directory = "../sample/ChileUni/NoSemi/GAMMA=5000000/";
%training_file = [directory 'chileDataL2R_highschool_nosemi_train.txt']
%model_file = [directory 'chileDataL2R_highschool_nosemi_model.m']


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TREC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%directory = "../sample/TREC/GAMMA=0/";
%training_file = [directory "features_with_total_order-zscore-train.csv"];
%model_file = [directory "features_with_total_order-zscore_model.m"];

%directory = "../sample/TREC/GAMMA=750/";
%training_file = [directory "features_with_total_order-zscore-train.csv"];
%model_file = [directory "features_with_total_order-zscore_model.m"];

%directory = "../sample/TREC/GAMMA=1500/";
%training_file = [directory "features_with_total_order-zscore-train.csv"];
%model_file = [directory "features_with_total_order-zscore_model.m"];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TREC BIG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%directory = "../sample/TREC-BIG/GAMMA=0/"
%training_file = [directory "features_with_total_order-withGender_withZscore_train.csv"];
%model_file = [directory "features_with_total_order-withGender_withZscore__model.m"];

%directory = "../sample/TREC-BIG/GAMMA=15000/"
%training_file = [directory "features_with_total_order-withGender_withZscore_train.csv"];
%model_file = [directory "features_with_total_order-withGender_withZscore__model.m"];

%directory = "../sample/TREC-BIG/GAMMA=75000/"
%training_file = [directory "features_with_total_order-withGender_withZscore_train.csv"];
%model_file = [directory "features_with_total_order-withGender_withZscore__model.m"];


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

directory = arg_list{1,1}
training_file = arg_list{2,1}
model_file = arg_list{3,1}
GAMMA = str2num(arg_list{4,1})

%GAMMA = 100000

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
disp(sprintf('training, %d iteration, %d examples, learning rate %f, gamma %d, ...\n', T, size(X,1), e, GAMMA))
tic();
omega = trainNN(GAMMA, directory, list_id, X, y, T, e);
training_time = toc();
disp(sprintf('finished training, time elapsed: %d seconds', training_time))
save(model_file, "omega");
