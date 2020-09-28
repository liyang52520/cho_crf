# -*- coding: utf-8 -*-

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_in, n_out, activation=None):
        super(MLP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)

        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_out={self.n_out}"
        s += ')'

        return s

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
