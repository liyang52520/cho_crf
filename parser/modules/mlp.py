# -*- coding: utf-8 -*-

import torch.nn as nn

from parser.modules.dropout import SharedDropout


class MLP(nn.Module):

    def __init__(self, n_layers, n_in, n_out, dropout=0, activation=None):
        super(MLP, self).__init__()
        self.n_layers = n_layers

        self.n_in = n_in
        self.n_mid = n_in // 2
        self.n_out = n_out
        # 对于bigram，尝试通过两层来压缩信息看看
        if self.n_layers == 2:
            self.linear_1 = nn.Linear(n_in, self.n_mid)
            self.linear_2 = nn.Linear(self.n_mid, n_out)
            self.activation_1 = nn.LeakyReLU(negative_slope=0.1)
        elif self.n_layers == 1:
            self.linear_1 = nn.Linear(n_in, n_out)
        else:
            raise Exception("Unexpected n_layers of MLP layer, expect 1 or 2.")

        if activation is None:
            self.activation_2 = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.activation_2 = activation
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'

        return s

    def reset_parameters(self):
        if self.n_layers == 2:
            nn.init.orthogonal_(self.linear_2.weight)
            nn.init.zeros_(self.linear_2.bias)
        nn.init.orthogonal_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)

    def forward(self, x):
        if self.n_layers == 1:
            x = self.linear_1(x)
            x = self.activation_2(x)
        else:
            x = self.linear_1(x)
            x = self.activation_1(x)
            x = self.dropout(x)
            x = self.linear_2(x)
            x = self.activation_2(x)

        return x
