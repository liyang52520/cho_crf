# -*- coding: utf-8 -*-

from collections import Counter

import torch

from parser.utils.fn import half_width, pad
from parser.utils.vocab import Vocab


class RawField(object):

    def __init__(self, name, fn=None):
        super(RawField, self).__init__()

        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        return sequence

    def transform(self, sequences):
        return [self.preprocess(sequence) for sequence in sequences]


class Field(RawField):

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, to_half_width=False, use_vocab=True, tokenize=None, fn=None):
        super(Field, self).__init__(name, fn)
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.to_half_width = to_half_width
        self.use_vocab = use_vocab
        self.tokenize = tokenize

        self.specials = [token for token in [pad, unk, bos, eos]
                         if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        if self.to_half_width:
            params.append(f"tohalfwidth={self.to_half_width}")
        s += f", ".join(params)
        s += f")"

        return s

    @property
    def vocab_size(self):
        if "_vocab_size" in self.__dict__:
            return self._vocab_size
        elif self.use_vocab:
            return self.vocab.n_init
        else:
            return 0

    @property
    def pad_index(self):
        return self.specials.index(self.pad) if self.pad is not None else 0

    @property
    def unk_index(self):
        return self.specials.index(self.unk) if self.unk is not None else 0

    @property
    def bos_index(self):
        return self.specials.index(self.bos) if self.bos is not None else None

    @property
    def eos_index(self):
        return self.specials.index(self.eos) if self.eos is not None else None

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]
        if self.to_half_width:
            sequence = [half_width(token) for token in sequence]

        return sequence

    def build(self, corpus, min_freq=1, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter(token
                          for sequence in sequences
                          for token in self.preprocess(sequence))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def decode(self, sequence):
        return self.vocab.id2token(sequence)

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class SubwordField(Field):
    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super().__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter()

        for sequence in sequences:
            for token in sequence:
                for piece in self.preprocess(token):
                    counter.update(piece)

        # vocab
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        # embed
        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors

    def transform(self, sequences):
        """

        Args:
            sequences list(list of tokens):

        Returns:

        """
        # if use vocab, transform token to id
        sequences = list(sequences)
        if self.use_vocab:
            sequences = [[[self.vocab[i] for i in token] if token else [self.unk] for token in seq]
                         for seq in sequences]

        # bos
        if self.bos:
            sequences = [[[self.bos_index]] + sequence
                         for sequence in sequences]

        # eos
        if self.eos:
            sequences = [sequence + [[self.eos_index]]
                         for sequence in sequences]

        # pad，这里的填充，只是将每一个句子里面的char，大小一致即可
        sequences = [pad([torch.tensor(ids) for ids in seq], self.pad_index)
                     for seq in sequences]

        return sequences
