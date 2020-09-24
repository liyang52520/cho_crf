#!/bin/zsh

nohup python -u run.py train --preprocess --device 2 --dir exp/ctb6.minus_span --ftrain data/pos/ctb6/train.conll --fdev data/pos/ctb6/dev.conll --ftest data/pos/ctb6/test.conll > results/ctb6.minus_span 2>&1 &