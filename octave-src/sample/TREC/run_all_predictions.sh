#!/bin/bash
# runs all predictions for Chile University data and saves results into respective folders

GIT_ROOT="$(git rev-parse --show-toplevel)"

PATH_TO_EXECUTABLE_DELTR=$GIT_ROOT/octave-src/src
PATH_TO_EXECUTABLE_LISTNET=$GIT_ROOT/Cao_src/listnet-master/src
PATH_TO_BIG_TREC_DATASETS=$GIT_ROOT/octave-src/sample/TREC #-BIG

echo ""
echo "################################# PREDICTING GENDER NOSEMI #############################################"

FOLD=fold_1

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/
#
#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/

FOLD=fold_2

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/

FOLD=fold_3

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/



FOLD=fold_4

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/


FOLD=fold_5

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/


FOLD=fold_6

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/



