#!/bin/zsh

nohup python -u run.py train --preprocess --embedding --device 0  --dir exp/ctb5.improve_3 --ftrain data/pos/ctb5/train.conll --fdev data/pos/ctb5/dev.conll --ftest data/pos/ctb5/test.conll > results/ctb5.improve_3 2>&1 &