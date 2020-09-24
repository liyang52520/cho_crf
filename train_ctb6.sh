#!/bin/zsh

nohup python -u run.py train --label-ngram 2 --label-smoothing --preprocess --device 7 --dir exp/ctb6.char --ftrain data/pos/ctb6/train.conll --fdev data/pos/ctb6/dev.conll --ftest data/pos/ctb6/test.conll > results/ctb6.char 2>&1 &