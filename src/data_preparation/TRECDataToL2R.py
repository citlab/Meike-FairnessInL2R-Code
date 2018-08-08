'''
Created on Aug 7, 2018

@author: mzehlike
'''

import pandas as pd

data = pd.read_csv('../../octave-src/sample/TREC-BIG/features_withListNetFormat_withGender_withZscore_candidateAmount-200_total.csv',
                   sep=',',
                   names=['query_id', 'gender', '1', '2', '3', '4', '5', 'score'])

test_queries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_fold1 = data.loc[~data['query_id'].isin(test_queries)]
test_fold1 = data.loc[data['query_id'].isin(test_queries)]

train_fold1.to_csv('../../octave-src/sample/TREC-BIG/fold_1/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
test_fold1.to_csv('../../octave-src/sample/TREC-BIG/fold_1/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

test_queries = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
train_fold2 = data.loc[~data['query_id'].isin(test_queries)]
test_fold2 = data.loc[data['query_id'].isin(test_queries)]

train_fold2.to_csv('../../octave-src/sample/TREC-BIG/fold_2/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
test_fold2.to_csv('../../octave-src/sample/TREC-BIG/fold_2/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

test_queries = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
train_fold3 = data.loc[~data['query_id'].isin(test_queries)]
test_fold3 = data.loc[data['query_id'].isin(test_queries)]

train_fold3.to_csv('../../octave-src/sample/TREC-BIG/fold_3/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
test_fold3.to_csv('../../octave-src/sample/TREC-BIG/fold_3/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

test_queries = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
train_fold4 = data.loc[~data['query_id'].isin(test_queries)]
test_fold4 = data.loc[data['query_id'].isin(test_queries)]

train_fold4.to_csv('../../octave-src/sample/TREC-BIG/fold_4/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
test_fold4.to_csv('../../octave-src/sample/TREC-BIG/fold_4/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)

test_queries = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
train_fold5 = data.loc[~data['query_id'].isin(test_queries)]
test_fold5 = data.loc[data['query_id'].isin(test_queries)]

train_fold5.to_csv('../../octave-src/sample/TREC-BIG/fold_5/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv', index=False, header=False)
test_fold5.to_csv('../../octave-src/sample/TREC-BIG/fold_5/features_withListNetFormat_withGender_withZscore_candidateAmount-200_test.csv', index=False, header=False)




