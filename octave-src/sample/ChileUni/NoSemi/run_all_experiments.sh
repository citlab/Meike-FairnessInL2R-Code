#!/bin/bash
# runs all experiments for Chile University data and saves result models into respective folders

GIT_ROOT="$(git rev-parse --show-toplevel)"

PATH_TO_EXECUTABLE_DELTR=$GIT_ROOT/octave-src/src
PATH_TO_EXECUTABLE_LISTNET=$GIT_ROOT/Cao_src/listnet-master/src
PATH_TO_CHILE_NOSEMI_DATASETS=$GIT_ROOT/octave-src/sample/ChileUni/NoSemi

GAMMA_SMALL=1000
GAMMA_LARGE=50000

echo $GIT_ROOT

##########################################################################################
# all gender experiments, no semi-private highschools
##########################################################################################

EXPERIMENT=gender

echo ""
echo "################################# RUNNING GENDER NOSEMI #############################################"

FOLD=fold_1

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD GAMMA=0..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD GAMMA=SMALL..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_2

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_3

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_4

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_5

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE


##########################################################################################
# all highschool type experiments, no semi-private highschools
##########################################################################################

EXPERIMENT=highschool

FOLD=fold_1

echo ""
echo "################################# RUNNING HIGHSCHOOL NOSEMI #############################################"

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_2

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_3

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_4

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

FOLD=fold_5

echo "$FOLD colorblind running..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/COLORBLIND/model.m

echo "$FOLD gamma=0 running..."
cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

echo "$FOLD gamma=small running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD gamma=large running..."
./train.m $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE


