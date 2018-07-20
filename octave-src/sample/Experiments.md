# Experiments

## TREC-BIG

| T | Gamma | run | done | pred | repo | eval | comments |
| --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 50,000 | 30.06.2018 |  |  |  |  | |
| 1000 | 1,000,000 | 29.06.2018 | 30.06.2018 |  |  |  | |
| 1000 | 5,000,000 | 28.06.2018 | 29.06.2018 |  |  |  | trying with dataset with continuous scores, using 50 queries for training, but only 200 candidates --> gamma too large, convergence looked weird |
| 3000 | 5,000,000 | x | 14.6.2018 | x | x | x | higher iterations did not make women to be distributed evenly, but also rates all women to top positions |
| 3000 | 100,000 | x | 14.6.2018 | x | x | x | |
| 1000 | 5,000,000 | x | x | x | x | x | made all women appear in top positions, super weird, trying to have better convergence |
| 1000 | 0 | x | x | x | x | x | |
| 1000 | Colorblind | x | x | x | x | x | |

## LSAT

### Gender

| T | Gamma | pid | run | done | pred | repo | eval | comments |
| --- | --- | --- | ---| --- | --- | --- | --- |
| 1000 | 5,000,000 | 1634 | 20.07.2018 | | | | | subsampled dataset, now has ~1700 candidates in training set |
| 1000 | 0 | 10417 | 20.07.2018 | | | | | subsampled dataset, now has ~1700 candidates in training set |
| 3000 | Colorblind | | | | | | |
