# -*- coding: utf-8 -*-

import argparse
import os

import torch

from parser.cmds import Evaluate, Train
from parser.utils.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Cho CRF Parser model.'
    )
    # sub parsers
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'train': Train()
    }

    for name, subcommand in subcommands.items():
        # special arguments
        subparser = subcommand.add_subparser(name, subparsers)
        # common arguments
        subparser.add_argument('--config', default='config.ini',
                               help='path to config file')
        subparser.add_argument('--dir', default='exp/ctb51.char',
                               help='path to saved files')
        subparser.add_argument('--preprocess', action='store_true',
                               help='whether to preprocess the data first')
        subparser.add_argument('--device', default='-1',
                               help='ID of GPU to use')
        subparser.add_argument('--seed', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=8, type=int,
                               help='max num of threads')
        subparser.add_argument('--feat', default='char',
                               choices=[None, 'char'],
                               help='choices of additional features')
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--label-ngram', default=1, type=int,
                               help='label ngram')
        subparser.add_argument('--buckets', default=32, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--label-smoothing', action='store_true',
                               help='whether to use label smoothing')
        subparser.add_argument('--embedding', action='store_true',
                               help='whether to use pretrained embedding')

    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.fields = os.path.join(args.dir, 'fields')
    args.model = os.path.join(args.dir, 'model')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = Config(args.config).update(vars(args))

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)
