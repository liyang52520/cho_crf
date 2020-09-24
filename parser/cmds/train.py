# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.metric import Metric


class Train(CMD):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--ftrain', default='data/pos/ctb5/train.conll',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/pos/ctb5/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/pos/ctb5/test.conll',
                               help='path to test file')
        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)

        train = Corpus.load(args.ftrain, self.fields)
        dev = Corpus.load(args.fdev, self.fields)
        test = Corpus.load(args.ftest, self.fields)

        train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields, args.buckets)
        test = TextDataset(test, self.fields, args.buckets)

        # set the data loaders
        train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        test.loader = batchify(test, args.batch_size)

        print(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'dev:':6} {len(dev):5} sentences, "
              f"{len(dev.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'test:':6} {len(test):5} sentences, "
              f"{len(test.loader):3} batches, "
              f"{len(train.buckets)} buckets")

        print("Create the model")

        # load pretrained
        embed = self.WORD.embed
        self.model = Model(args)
        if args.embedding:
            self.model = Model(args).load_pretrained(embed)
        print(f"{self.model}\n")

        # to device
        self.model = self.model.to(args.device)
        # ?
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        # optimizer
        self.optimizer = AdamW(self.model.parameters(),
                               args.lr,
                               (args.mu, args.nu),
                               args.epsilon,
                               args.weight_decay)
        # scheduler
        decay_steps = args.decay_epochs * len(train.loader)
        self.scheduler = ExponentialLR(self.optimizer,
                                       args.decay ** (1 / decay_steps))

        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch} / {args.epochs}:")
            start = datetime.now()
            # train
            self.train(self.model, train.loader, self.optimizer, self.scheduler)
            # dev
            loss, dev_metric = self.evaluate(self.model, dev.loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric and epoch > args.patience // 10:
                best_e, best_metric = epoch, dev_metric
                if hasattr(self.model, 'module'):
                    self.model.module.save(args.model)
                else:
                    self.model.save(args.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            if hasattr(self.model, 'module'):
                self.model.module.save(os.path.join(args.dir, 'model.last_epoch'))
            else:
                self.model.save(os.path.join(args.dir, 'model.last_epoch'))

            total_time += t

            # patience
            if epoch - best_e >= args.patience:
                break

        # test
        self.model = Model.load(args.model)
        loss, metric = self.evaluate(self.model, test.loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def train(self, model, loader, optimizer, scheduler):
        model.train()

        for data in loader:
            if self.args.feat == 'char':
                words, chars, labels = data
                feed_dict = {"words": words, "chars": chars}
            else:
                words, labels = data
                feed_dict = {"words": words}

            optimizer.zero_grad()

            # words: [batch_size, seq_len + 1] (因为加了<bos>)
            # mask: [batch_size, seq_len]
            mask = words.ne(self.args.pad_index)[:, 1:]

            # emits: [batch_size, seq_len, n_labels, n_labels]
            emits = model(feed_dict)

            # compute crf loss
            loss = model.loss(emits, labels, mask)
            loss.backward()
            #
            nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()
            scheduler.step()
