#!/bin/zsh

nohup python -u run.py train --preprocess --embedding --device 2  --dir exp/ctb5.combine --ftrain data/pos/ctb5/train.conll --fdev data/pos/ctb5/dev.conll --ftest data/pos/ctb5/test.conll > results/ctb5.combine 2>&1 &