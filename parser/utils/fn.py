# -*- coding: utf-8 -*-

import unicodedata


def half_width(token):
    return unicodedata.normalize('NFKC', token)


def pad(tensors, padding_value=0):
    size = [len(tensors)] + [max(tensor.n_labels(i) for tensor in tensors)
                             for i in range(len(tensors[0].n_labels()))]
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(0, i) for i in tensor.n_labels()]] = tensor
    return out_tensor
