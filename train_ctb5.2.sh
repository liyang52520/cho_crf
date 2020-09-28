#!/bin/zsh

nohup python -u run.py train --label-ngram 2 --embedding --preprocess --device 1  --dir exp/ctb5.master.2 --ftrain data/pos/ctb5/train.conll --fdev data/pos/ctb5/dev.conll --ftest data/pos/ctb5/test.conll > results/ctb5.master.2 2>&1 &