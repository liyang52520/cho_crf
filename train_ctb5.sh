#!/bin/zsh

nohup python -u run.py train --label-ngram 2 --label-smoothing --preprocess --device 6  --dir exp/ctb5.char --ftrain data/pos/ctb5/train.conll --fdev data/pos/ctb5/dev.conll --ftest data/pos/ctb5/test.conll > results/ctb5.char 2>&1 &