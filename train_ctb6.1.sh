#!/bin/zsh

nohup python -u run.py train --label-ngram 1 --embedding --preprocess --device 2 --dir exp/ctb6.master.1 --ftrain data/pos/ctb6/train.conll --fdev data/pos/ctb6/dev.conll --ftest data/pos/ctb6/test.conll > results/ctb6.master.1 2>&1 &