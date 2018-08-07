#!/usr/bin/octave -qf
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
%training_file = arg_list{1,1};
%model_file = arg_list{2,1};
%training_file = '../sample/sample_training_data.m'
%model_file = '../sample/sample_model.m'
%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/Semi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/NoSemi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_nosemi_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/NoSemi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_nosemi_model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-train.csv'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC-BIG/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC-BIG/COLORBLIND_GAMMA=0/model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/gender/LawStudents_Gender_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/gender/COLORBLIND/model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_asian/LawStudents_Race_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_asian/COLORBLIND/model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_black/LawStudents_Race_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_black/COLORBLIND/model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_hispanic/COLORBLIND/model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_mexican/LawStudents_Race_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_mexican/COLORBLIND/model.m'

%training_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_train.txt'
%model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_puertorican/COLORBLIND/model.m'

FEAT_START = 3

directory = arg_list{1,1}
training_file = arg_list{2,1}
model_file = arg_list{3,1}


% load constants
addpath(".")
source "./global.m";

% load training dataset
disp('loading training data...')
data = load(training_file);
list_id = data(:,1);
X = data(:,FEAT_START:size(data,2)-1);
y = data(:,size(data,2));

% launch the training routine
disp(sprintf('training, %d iteration, %d examples, learning rate %f...', T, size(X,1), e))
tic();
omega = trainNN(list_id, directory, X, y, T, e);
training_time = toc();
disp(sprintf('finished training, time elapsed: %d seconds', training_time))
save(model_file, "omega");
