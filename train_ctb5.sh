#!/bin/zsh

nohup python -u run.py train --preprocess --embedding --device 1  --dir exp/ctb5.minus_span --ftrain data/pos/ctb5/train.conll --fdev data/pos/ctb5/dev.conll --ftest data/pos/ctb5/test.conll > results/ctb5.minus_span 2>&1 &