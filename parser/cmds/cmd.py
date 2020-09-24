import os

import torch

from parser.utils import Embedding
from parser.utils.common import bos, unk, pad
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import SubwordField, Field
from parser.utils.metric import LabelMetric


class CMD(object):
    def __call__(self, args):
        self.args = args
        # create dir for files
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)

        # fields
        if args.preprocess or not os.path.exists(args.fields):
            print("Preprocess the data")

            # word field
            self.WORD = Field('words',
                              bos=bos, eos=None, pad=pad, unk=unk,
                              lower=True)

            # label field
            self.LABEL = Field('labels', bos=bos, eos=None, pad=pad)

            # feat field（代码中只有char，也不用别的了，删掉了其余特征）
            if args.feat == "char":
                self.CHAR = SubwordField("chars", fix_len=args.fix_len,
                                         bos=bos, eos=None, pad=pad, unk=unk, lower=True)
                self.fields = CoNLL(WORD=(self.WORD, self.CHAR), LABEL=self.LABEL)
            else:
                self.fields = CoNLL(WORD=self.WORD, LABEL=self.LABEL)

            # load dataset
            train = Corpus.load(args.ftrain, self.fields)

            # build vocab
            embed = None
            if args.embedding:
                embed = Embedding.load('data/embedding/giga.100.txt', )
            self.WORD.build(train, args.min_freq, embed)
            self.LABEL.build(train)
            if args.feat == "char":
                self.CHAR.build(train)
            # save fields
            torch.save(self.fields, args.fields)
        else:
            # load fields
            self.fields = torch.load(args.fields)
            if args.feat == 'char':
                self.WORD, self.CHAR = self.fields.WORD
            else:
                self.WORD = self.fields.WORD
            self.LABEL = self.fields.LABEL

        args.update({
            'n_words': self.WORD.vocab_size,
            'n_labels': self.LABEL.vocab_size,
            'pad_index': self.WORD.pad_index,
            'unk_index': self.WORD.unk_index,
            'label_bos_index': self.LABEL.bos_index,
            'label_pad_index': self.LABEL.pad_index
        })

        vocab = f"{self.WORD}\n{self.LABEL}\n"
        if hasattr(self, 'CHAR'):
            args.update({
                'n_chars': self.CHAR.vocab_size,
            })
            vocab += f"{self.CHAR}\n"

        print(f"Override the default configs\n{args}")
        print(vocab[:-1])

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()

        total_loss = 0
        metric = LabelMetric()

        for data in loader:
            if self.args.feat == 'char':
                words, chars, labels = data
                feed_dict = {"words": words, "chars": chars}
            else:
                words, labels = data
                feed_dict = {"words": words}

            # mask
            mask = words.ne(self.args.pad_index)[:, 1:]

            # 计算分值
            emits = model(feed_dict)

            # loss
            loss = model.loss(emits, labels, mask)
            total_loss += loss.item()

            # predict
            predicts = model.predict(emits, mask)
            labels = labels[:, 1:]
            metric(predicts, labels, mask)
        total_loss /= len(loader)

        return total_loss, metric
