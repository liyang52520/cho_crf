# -*- coding: utf-8 -*-

import torch.nn as nn

from parser.modules.dropout import SharedDropout


class MinusSpanMLP(nn.Module):

    def __init__(self, n_in, n_mid, n_out, dropout=0, activation=None):
        super(MinusSpanMLP, self).__init__()

        self.n_in = n_in
        self.n_mid = n_mid
        self.n_out = n_out
        self.linear_1 = nn.Linear(n_in, n_mid)
        if activation is None:
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.activation = activation

        self.dropout = SharedDropout(p=dropout)

        self.linear_2 = nn.Linear(n_mid, n_out, bias=False)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_mid={self.n_mid}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'

        return s

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear_1.weight)
        nn.init.orthogonal_(self.linear_2.weight)
        nn.init.zeros_(self.linear_1.bias)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x
