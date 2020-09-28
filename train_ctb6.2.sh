#!/bin/zsh

nohup python -u run.py train --label-ngram 2 --preprocess --device 2 --dir exp/ctb6.master.2 --ftrain data/pos/ctb6/train.conll --fdev data/pos/ctb6/dev.conll --ftest data/pos/ctb6/test.conll > results/ctb6.master.2 2>&1 &