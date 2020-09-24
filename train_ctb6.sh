#!/bin/zsh

nohup python -u run.py train --preprocess --device 7 --dir exp/ctb6.combine --ftrain data/pos/ctb6/train.conll --fdev data/pos/ctb6/dev.conll --ftest data/pos/ctb6/test.conll > results/ctb6.combine 2>&1 &