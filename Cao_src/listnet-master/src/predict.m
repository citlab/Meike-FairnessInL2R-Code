#!/usr/bin/env octave
% command line arguments:
% predict.m model feature_file

% suppress output
more off;


% load constants
addpath(".")
source "./globals.m";

arg_list = argv()

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_test.txt';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/NoSemi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_nosemi_model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/ChileUni/NoSemi/COLORBLIND_GAMMA=0/chileDataL2R_colorblind_nosemi_test.txt';

%omega = load(model_file = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC/COLORBLIND_GAMMA=0/features_with_total_order-zscore-test.csv';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC-BIG/COLORBLIND_GAMMA=0/model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/TREC-BIG/features_withListNetFormat_withGender_withZscore_test.csv';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/gender/COLORBLIND/model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/gender/LawStudents_Gender_test.txt';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_asian/COLORBLIND/model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_asian/LawStudents_Race_test.txt';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_black/COLORBLIND/model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_black/LawStudents_Race_test.txt';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_hispanic/COLORBLIND/model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_test.txt';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_mexican/COLORBLIND/model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_mexican/LawStudents_Race_test.txt';

%omega = load('/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_puertorican/COLORBLIND/model.m');
%drgfile = '/home/mzehlike/workspace/Meike-FairnessInL2R-Code/octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_test.txt';

FEAT_START = 2

test_file = arg_list{1,1}
model_file = arg_list{2,1}
output_dir = arg_list{3,1}

omega = load(model_file);
drg = load(test_file);

list_id = drg(:,1);
X = drg(:,FEAT_START:size(drg,2)-1);

%omega_values = omega.omega(1:size(omega),1);
omega_values = omega.omega(:);

z =  X * omega_values;
doc_ids = 1:size(z);

# also write y for later evaluation
y = drg(:, size(drg,2));
y = [list_id, doc_ids', y];

filename = [output_dir "trainingScores_ORIG.pred"];
dlmwrite(filename, y);

# add document ids for later evaluation
z = [list_id, doc_ids', z];

unsorted_ranks = z;
filename = [output_dir "predictions_UNSORTED.pred"];
dlmwrite(filename, unsorted_ranks);

# add a little random to avoid ties
r = @(i) (i+rand*0.02-0.01);

for id = unique(list_id)'
    indexes = find(list_id==id);
    z_temp = z(indexes, :);
    z(indexes, :) = sortrows(z_temp, -3);
endfor
sorted_ranks = z;
filename = [output_dir "predictions_SORTED.pred"];

dlmwrite(filename, sorted_ranks)
%figure(); plot(z);
