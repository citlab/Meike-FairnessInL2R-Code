#!/bin/bash
# runs all experiments for Chile University data and saves result models into respective folders

PATH_TO_EXECUTABLE_DELTR=../../../src/train.m
PATH_TO_EXECUTABLE_LISTNET=../../../../Cao_src/listnet-master/src/train.m

##########################################################################################
# all gender experiments, no semi-private highschools
##########################################################################################

EXPERIMENT=gender

echo "gender Experiment"

FOLD=fold_1

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_2

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_3

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_4

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_5

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000


##########################################################################################
# all highschool type experiments, no semi-private highschools
##########################################################################################

EXPERIMENT=highschool

FOLD=fold_1

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_2

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold2_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_3

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold3_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_4

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold4_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000

FOLD=fold_5

echo "$FOLD colorblind running..."
$PATH_TO_EXECUTABLE_LISTNET $EXPERIMENT/$FOLD/COLORBLIND/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/COLORBLIND/model.m
echo "$FOLD gamma=0 running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
echo "$FOLD gamma=small running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
echo "$FOLD gamma=large running..."
$PATH_TO_EXECUTABLE_DELTR $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_highschool_nosemi_fold5_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000


