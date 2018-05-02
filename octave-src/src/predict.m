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
omega = load('../sample/synthetic_score_gender/top_male_bottom_female/sample_model_gender_sep.m');
drgfile = '../sample/synthetic_score_gender/top_male_bottom_female/sample_test_data_scoreAndGender_separated.txt';
drg = load(drgfile);

list_id = drg(:,1);
X = drg(:,2:size(drg,2)-1);

z =  X * omega.omega;

# add a little random to avoid ties
r = @(i) (i+rand*0.02-0.01);

for id = unique(list_id)'
    indexes = find(list_id==id);
    a(indexes) = floor(ranks(arrayfun(r, z(indexes))));
    # add protection status to a for later evaluation
    prot = [X(a, PROT_COL)']
    a = [prot; a]
endfor
ranks = a';

dlmwrite(sprintf('%s.pred', drgfile), ranks)
