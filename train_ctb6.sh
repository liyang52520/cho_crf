#!/bin/zsh

nohup python -u run.py train --preprocess --embedding --device 4 --dir exp/ctb6.improve_3 --ftrain data/pos/ctb6/train.conll --fdev data/pos/ctb6/dev.conll --ftest data/pos/ctb6/test.conll > results/ctb6.improve_3 2>&1 &