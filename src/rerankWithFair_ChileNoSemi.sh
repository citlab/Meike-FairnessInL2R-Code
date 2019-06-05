#!/bin/bash
# rerank Chile SAT dataset folds using FA*IR for different values of p

python3 main.py --rerank engineering-NoSemi 0.1
python3 main.py --rerank engineering-NoSemi 0.2
python3 main.py --rerank engineering-NoSemi 0.3
python3 main.py --rerank engineering-NoSemi 0.4
python3 main.py --rerank engineering-NoSemi 0.5
python3 main.py --rerank engineering-NoSemi 0.6
python3 main.py --rerank engineering-NoSemi 0.7
python3 main.py --rerank engineering-NoSemi 0.8
python3 main.py --rerank engineering-NoSemi 0.9

