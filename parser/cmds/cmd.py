import os

import torch

from parser.utils import Embedding
from parser.utils.common import bos, eos, unk, pad
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
                              bos=bos if args.label_ngram > 1 else None,
                              eos=eos if args.label_ngram > 2 else None,
                              pad=pad, unk=unk, lower=True, to_half_width=True)

            # label field
            self.LABEL = Field('labels',
                               bos=bos if args.label_ngram > 1 else None,
                               eos=eos if args.label_ngram > 2 else None,
                               pad=pad)

            # feat field（代码中只有char，也不用别的了，删掉了其余特征）
            if args.feat == "char":
                self.CHAR = SubwordField("chars", fix_len=args.fix_len,
                                         bos=bos if args.label_ngram > 1 else None,
                                         eos=eos if args.label_ngram > 2 else None,
                                         pad=pad, unk=unk, lower=True, to_half_width=True)
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
            mask = words.ne(self.args.pad_index)
            if self.args.label_ngram > 1:
                mask = mask[:, (self.args.label_ngram - 1):]

            # 计算分值
            emits = model(feed_dict)

            # loss
            loss = model.loss(emits, labels, mask)
            total_loss += loss.item()

            # predict
            predicts = model.predict(emits, mask)
            if self.args.label_ngram > 1:
                # 去掉<bos>
                labels = labels[:, 1:]
            if self.args.label_ngram > 2:
                # 如果大于2，要去掉<bos>和<eos>，但是目前不会遇到
                pass
            metric(predicts, labels, mask)
        total_loss /= len(loader)

        return total_loss, metric
