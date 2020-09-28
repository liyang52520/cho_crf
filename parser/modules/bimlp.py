# -*- coding: utf-8 -*-

import torch.nn as nn

from parser.modules.dropout import SharedDropout


class BiMLP(nn.Module):

    def __init__(self, n_in, n_mid, n_out, dropout=0, activation=None):
        super(BiMLP, self).__init__()
        self.n_in = n_in
        self.n_mid = n_mid
        self.n_out = n_out

        # linear_1
        self.linear_1 = nn.Linear(n_in, self.n_mid)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        # linear_2
        self.linear_2 = nn.Linear(self.n_mid, n_out)
        if activation is None:
            self.activation_2 = nn.Identity()
        else:
            self.activation_2 = activation

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_mid={self.n_mid} n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'

        return s

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.orthogonal_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.activation_2(x)

        return x
