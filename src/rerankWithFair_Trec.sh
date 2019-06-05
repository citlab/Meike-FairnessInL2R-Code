#!/bin/bash
# rerank trec folds using FA*IR for different values of p

python3 main.py --rerank trec 0.1
python3 main.py --rerank trec 0.2
python3 main.py --rerank trec 0.3
python3 main.py --rerank trec 0.4
python3 main.py --rerank trec 0.5
python3 main.py --rerank trec 0.6
python3 main.py --rerank trec 0.7
python3 main.py --rerank trec 0.8
python3 main.py --rerank trec 0.9

