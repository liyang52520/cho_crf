#!/bin/zsh

nohup python -u run.py train--embedding --preprocess --device 2 --dir exp/ctb6.improve_4 --ftrain data/pos/ctb6/train.conll --fdev data/pos/ctb6/dev.conll --ftest data/pos/ctb6/test.conll > results/ctb6.improve_4 2>&1 &