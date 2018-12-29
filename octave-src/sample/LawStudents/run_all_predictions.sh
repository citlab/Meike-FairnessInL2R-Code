#!/bin/bash
# runs all predictions for Chile University data and saves results into respective folders


#################################
#
#   Created by Gina - Possibly the experiment runs, other than the PREPROCESSED ones, are not correct
#   Gamma is therefore set to zero !
#
##################################

GIT_ROOT="$(git rev-parse --show-toplevel)"

PATH_TO_EXECUTABLE_DELTR=$GIT_ROOT/octave-src/src
PATH_TO_EXECUTABLE_LISTNET=$GIT_ROOT/Cao_src/listnet-master/src
PATH_TO_LSAT_DATASETS=$GIT_ROOT/octave-src/sample/LawStudents

echo ""
echo "################################# PREDICTING GENDER #############################################"

FOLD=gender

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Gender_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/

echo ""
echo "################################# PREDICTING RACE #############################################"

FOLD=race_asian

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/

FOLD=race_black

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/



FOLD=race_hispanic

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/


FOLD=race_mexican

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/


FOLD=race_puertorican

echo "$FOLD predictions..."
cd $PATH_TO_EXECUTABLE_LISTNET
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/model.m $PATH_TO_LSAT_DATASETS/$FOLD/COLORBLIND/

./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/model.m $PATH_TO_LSAT_DATASETS/$FOLD/PREPROCESSED/

#cd $PATH_TO_EXECUTABLE_DELTR
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=0/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=SMALL/
#
#./predict.m $PATH_TO_LSAT_DATASETS/$FOLD/LawStudents_Race_test.txt $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/model.m $PATH_TO_LSAT_DATASETS/$FOLD/GAMMA\=LARGE/
#


