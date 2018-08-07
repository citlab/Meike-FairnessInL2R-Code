#!/bin/bash
# runs all experiments for Chile University data and saves result models into respective folders

PATH_TO_EXECUTABLE=../../../src/train.m

EXPERIMENT=gender

FOLD=fold_1

$PATH_TO_EXECUTABLE $EXPERIMENT/$FOLD/GAMMA\=0/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=0/model.m 0
$PATH_TO_EXECUTABLE $EXPERIMENT/$FOLD/GAMMA\=SMALL/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m 100000
$PATH_TO_EXECUTABLE $EXPERIMENT/$FOLD/GAMMA\=LARGE/ $EXPERIMENT/$FOLD/chileDataL2R_gender_nosemi_fold1_train.txt $EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m 5000000


