#!/bin/zsh

nohup python -u run.py train --label-ngram 1 --embedding --preprocess --device 0  --dir exp/ctb5.master.1 --ftrain data/pos/ctb5/train.conll --fdev data/pos/ctb5/dev.conll --ftest data/pos/ctb5/test.conll > results/ctb5.master.1 2>&1 &