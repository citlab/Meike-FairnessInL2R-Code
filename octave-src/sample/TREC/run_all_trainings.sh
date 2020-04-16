#!/bin/bash
# runs all trainings for big TREC data and saves result models into respective folders

export LD_PRELOAD=libGLX_mesa.so.0 	#very dirty hack to workaround this octave bug: error: __osmesa_print__: 
					#Depth and stencil doesn't match, are you sure you are using OSMesa >= 9.0?

GIT_ROOT="$(git rev-parse --show-toplevel)"

PATH_TO_EXECUTABLE_DELTR=$GIT_ROOT/octave-src/src
PATH_TO_EXECUTABLE_LISTNET=$GIT_ROOT/Cao_src/listnet-master/src
PATH_TO_BIG_TREC_DATASETS=$GIT_ROOT/octave-src/sample/TREC

GAMMA_SMALL=20000
GAMMA_LARGE=200000

echo ""
################################################################################

FOLD=Old_Datasets 

echo "$FOLD COLORBLIND..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m

echo "$FOLD PREPROCESSED..."
cd $PATH_TO_EXECUTABLE_LISTNET
./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m

./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/model.m

./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/model.m

echo "$FOLD GAMMA=0..."

cd $PATH_TO_EXECUTABLE_DELTR 
./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m 0


echo "$FOLD GAMMA=SMALL..."
./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

echo "$FOLD GAMMA=LARGE..."
./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

######################################################################

#FOLD=fold_1

#echo "$FOLD COLORBLIND..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m

#echo "$FOLD PREPROCESSED..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/model.m

#echo "$FOLD GAMMA=0..."

#cd $PATH_TO_EXECUTABLE_DELTR 
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m 0


#echo "$FOLD GAMMA=SMALL..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

#echo "$FOLD GAMMA=LARGE..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

###################################################################################

#FOLD=fold_2

#echo "$FOLD COLORBLIND..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m

#echo "$FOLD PREPROCESSED..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/model.m

#echo "$FOLD GAMMA=0..."
#cd $PATH_TO_EXECUTABLE_DELTR 
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m 0

#echo "$FOLD GAMMA=SMALL..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

#echo "$FOLD GAMMA=LARGE..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

#####################################################################################

#FOLD=fold_3

#echo "$FOLD COLORBLIND..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m

#echo "$FOLD PREPROCESSED..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/model.m

#echo "$FOLD GAMMA=0..."
#cd $PATH_TO_EXECUTABLE_DELTR 
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m 0

#echo "$FOLD GAMMA=SMALL..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

#echo "$FOLD GAMMA=LARGE..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

######################################################################################

#FOLD=fold_4

#echo "$FOLD COLORBLIND..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m

#echo "$FOLD PREPROCESSED..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/model.m

#echo "$FOLD GAMMA=0..."
#cd $PATH_TO_EXECUTABLE_DELTR 
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m 0

#echo "$FOLD GAMMA=SMALL..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

#echo "$FOLD GAMMA=LARGE..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

#######################################################################################

#FOLD=fold_5

#echo "$FOLD COLORBLIND..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/model.m

#echo "$FOLD PREPROCESSED..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/model.m

#echo "$FOLD GAMMA=0..."
#cd $PATH_TO_EXECUTABLE_DELTR 
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=0/model.m 0

#echo "$FOLD GAMMA=SMALL..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

#echo "$FOLD GAMMA=LARGE..."
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE

#######################################################################################

#FOLD=fold_6

#echo "$FOLD COLORBLIND..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/COLORBLIND/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/CO$

#echo "$FOLD PREPROCESSED..."
#cd $PATH_TO_EXECUTABLE_LISTNET
#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PPlus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PPlus/model.m

#./train.m $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_BIG_TREC_DATASETS/$FOLD/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_RERANKED_PMinus.csv $PATH_TO_BIG_TREC_DATASETS/$FOLD/PREPROCESSED_PMinus/model.m
