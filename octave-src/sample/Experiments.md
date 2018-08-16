# Experiments

## CHILE UNIVERSITY

### Gender (each of 5 folds running)

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 3000 | 10126 | 10.08.2018 | x | x | x | x | running on lsat |
| 1000 | 0 | 3811 | 07.08.2018 | x | x | x | x | running on lsat |
| 1000 | 1000 | 3811 | 07.08.2018 | x | x | x | x | running on lsat |
| 1000 | 50000 | 3811 | 07.08.2018 | x | x | x | x | running on lsat |
| 1000 | Colorblind | 3811 | 07.08.2018 | x | x | x | x | running on lsat |

### Highschool (each of 5 folds running)

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 3000 | 10126 | 10.08.2018 | x | x | x | x | running on lsat |
| 1000 | 0 | 3811 | 07.08.2018 | x | x | x | x | running on lsat |
| 1000 | 1000 | 3811 | 07.08.2018 | x | x | x | x | running on lsat |
| 1000 | 50000 | 3811 | 07.08.2018 | x | x | x | x | running on lsat, redo this one for fold 3 [Done]|
| 1000 | Colorblind | 3811 | 07.08.2018 | x | x | x | x | running on lsat |


## TREC-BIG

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 3000 | all | xxx | 09.08.2018 |  |  |  |  | fold 4-5, running on LSAT, Has same gammas as in fold 6, as is 20,000 and 200,000 |
| 3000 | all | 4649 | 07.08.2018 |  |  |  |  | fold 1-3, fold 6 is already done in former experiments. Has same gammas as in fold 6, as is 20,000 and 200,000 |
| 3000 | 20000 | 1634 | 26.07.2018 | x | x | x | x | running on lsat |
| 3000 | 0 | 1634 | 25.07.2018 | x | x | x | x | running on lsat |
| 3000 | 200,000 | 30551 | 25.07.2018 | x | x | x | x |  |
| 3000 | Colorblind | -- | 25.07.2018 | x | x | x | x |  |
| 1000 | 0 | -- | x | x | x | x | x | |
| 1000 | Colorblind | -- | x | x | x | x | x | |
| --- | --- | --- |  --- | ---| --- | --- | --- | --- |
| 1000 | 100,000 | -- | 24.07.2018 | x | x |  |  | FAILED: number of iterations too small, gamma small enough though |
| 1000 | 50,000 | -- | 30.06.2018 | 01.07.2018 | x |  |  | Gamma might be to large for the small gamma case, have to wait to find large gamma first|
| 1000 | 500,000 | -- | 23.07.2018 | x |  |  |  | FAILED: gamma too large, cost function looks weird; running on LSAT server with PID 31731 |
| 1000 | 1,000,000 | -- | 29.06.2018 | 30.06.2018 | x |  |  | FAILED; gamma still too large |
| 1000 | 5,000,000 | -- | 28.06.2018 | 29.06.2018 |  |  |  | trying with dataset with continuous scores, using 50 queries for training, but only 200 candidates --> gamma too large, convergence looked weird |
| 3000 | 5,000,000 | -- | x | 14.6.2018 | x | x | x | higher iterations did not make women to be distributed evenly, but also rates all women to top positions |
| 3000 | 100,000 | -- | x | 14.6.2018 | x | x | x | |
| 1000 | 5,000,000 | -- | x | x | x | x | x | made all women appear in top positions, super weird, trying to have better convergence |

## LSAT

### Gender

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 0 | 31731 | 24.07.2018 | x | x | x | x | x |
| 1000 | 1,000,000 | 31731 | 24.07.2018 | x | x | x | x | doesn't change a lot going back to 500,000 |
| 1000 | 500,000 | 31731 | 24.07.2018 | x | x | x | x | might be a bit too small |
| 1000 | 2,000,000 | 30341 | 24.07.2018 | x | x | x | x |  |
| 1000 | Colorblind | -- | 24.07.2018 | x | x | x | x | |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 1,000,000 | 1399 | 23.07.2018 | x | x | | | gamma seems to be still too small, because mean of positions of protected and non-protected group is not yet equal|
| 1000 | 5,000,000 | 1634 | 20.07.2018 | 21.07.2018 | x | | | FAILED; subsampled dataset, now has ~1700 candidates in training set |
| 1000 | 0 | 10417 | 20.07.2018 | 21.07.2018 | x | | | FAILED; subsampled dataset, now has ~1700 candidates in training set |



### Asian

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 500,000 | 30341 | 25.07.2018 | x | x | x | x |  |
| 1000 | 2,000,000 | 30341 | 24.07.2018 | x | x | x | x |  |
| 1000 | 0 | 1634 | 23.07.2018 | x | x | x | x |  |
| 1000 | Colorblind | -- | 23.07.2018 | x | x | x | x | |


### Black

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 50,000 | 9076 | 16.08.2018 |  |  |  |  |  |
| 1000 | 500,000 | 10417 | 25.07.2018 | x | x | x | x |  |
| 1000 | 2,000,000 | 10417 | 24.07.2018 | x | x | x | x |  |
| 1000 | 0 | 10417 | 23.07.2018 | x | x | x | x |  |
| 1000 | Colorblind | -- | 23.07.2018 | x | x | x | x |


### Hispanic

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 500,000 | 1634 | 25.07.2018 | x | x | x | x |  |
| 1000 | 2,000,000 | 1399 | 24.07.2018 | x | x | x | x | running on TREC server |
| 1000 | 0 | 1399 | 23.07.2018 | x | x | x | x | running on TREC server |
| 1000 | Colorblind | -- | 23.07.2018 | x | x | x | x |


### Mexican

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 500,000 | 24011 | 25.07.2018 | x | x | x | x |  |
| 1000 | 2,000,000 | 29541 | 24.07.2018 | x | x | x | x | running on TREC server |
| 1000 | 0 | 29541 | 23.07.2018 | x | x | x | x | running on TREC server |
| 1000 | Colorblind | -- | 23.07.2018 | x | x | x | x |



### Puertorican

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 500,000 | 31731 | 25.07.2018 | x | x | x | x |  |
| 1000 | 2,000,000 | 4649 | 24.07.2018 | x | x | x | x | running on TREC server |
| 1000 | 0 | 30551 | 23.07.2018 | x | x | x | x | running on TREC server |
| 1000 | Colorblind | -- | 23.07.2018 | x | x | x | x | |

