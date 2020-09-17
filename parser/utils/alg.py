# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


@torch.enable_grad()
def crf(emits, mask, target=None, marg=False, order=1, label_bos_index=-1):
    """

    Args:
        emits (torch.Tensor): [batch_size, seq_len, n_labels]
            发射分值
        mask (torch.Tensor): [batch_size, seq_len]
            掩码矩阵，非<pad>为True
        target (torch.Tensor): [batch_size, seq_len]
            gold labels (Default: None)
        marg (bool):
            是否计算边缘概率
        order (int):
            crf阶数
        label_bos_index (int):
            当使用bi gram时，我们需要bos来计算初始分值

    Returns:

    """
    total_token = mask.sum().float()
    # 改变emits，target和mask的维度
    # emits: [seq_len, batch_size, n_labels]
    emits = emits.transpose(0, 1).requires_grad_()
    target = target.t() if target is not None else None
    # mask: [seq_len, batch_size]
    mask = mask.t()

    # log_z
    log_z = compute_log_z(emits, mask, order=order, label_bos_index=label_bos_index).sum()

    # probs，没懂这是啥
    training = emits.requires_grad
    probs = emits.transpose(0, 1)
    if marg:
        probs, = autograd.grad(log_z, emits, retain_graph=training)
        probs = probs.transpose(0, 1)
    if target is None:
        return probs

    # scores
    # torch.gather([seq_len, batch_size, n_labels], -1, [seq_len, batch_size, 1]).squeeze(-1)
    # => [seq_len, batch_size] => [n]
    scores = emits.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).masked_select(mask).sum()

    # loss
    loss = (log_z - scores) / total_token

    return loss, probs


def compute_log_z(emits, mask, order, label_bos_index=-1):
    """

    Args:
        emits (torch.Tensor): [batch_size, seq_len, n_labels]
            发射分值
        mask (torch.Tensor): [batch_size, seq_len]
            掩码矩阵，非<pad>为True
        order (int):
            crf阶数
        label_bos_index (int):

    Returns:

    """

    seq_len, batch_size, n_labels = emits.shape

    # 因为double精度更高吗?
    emits = emits.double()

    # 计算原来label的个数
    n_origin_labels = int(n_labels ** (1 / order))

    # log_alpha: [batch_size, n_origin_label]
    log_alpha = emits.new_zeros(batch_size, n_origin_labels, dtype=torch.double)

    # 计算log_alpha的第一个状态
    if order == 1:
        # 初始为发射到第一个词的分值（uni_gram不需要转移分值）
        # emits: [seq_len, batch_size, n_origin_labels]
        log_alpha += emits[0, :]
    elif order == 2:
        # emits: [seq_len, batch_size, n_origin_labels, n_origin_labels]
        emits = emits.view(seq_len, batch_size, n_origin_labels, n_origin_labels)
        log_alpha += emits[0, :, label_bos_index, :]

    # 计算剩余状态
    for i in range(1, seq_len):
        # 对于1阶
        # scores: [batch_size, n_origin_labels, 1] + [batch_size, 1, n_origin_labels]
        #        => [batch_size, n_origin_labels, n_origin_labels]
        if order == 1:
            scores = log_alpha.unsqueeze(-1) + emits[i].unsqueeze(1)
        # 对于2阶：
        # scores: [batch_size, n_origin_labels, 1] + [batch_size, n_origin_labels, n_origin_labels]
        elif order == 2:
            scores = log_alpha.unsqueeze(-1) + emits[i]
        else:
            raise Exception("wrong CRF order, excepted in {1, 2}")

        # temp_log_alpha: [batch_size, n_origin_labels]
        temp_log_alpha = torch.logsumexp(scores, dim=1)

        # 根据mask更新log_alpha
        log_alpha[mask[i]] = temp_log_alpha[mask[i]]

    # log_z: [batch_size]
    log_z = torch.logsumexp(log_alpha, dim=1)

    return log_z.sum()


def viterbi(emits, mask, order, label_bos_index):
    if order == 1:
        return viterbi1o(emits, mask)
    elif order == 2:
        return viterbi2o(emits, mask, label_bos_index)
    else:
        raise Exception("wrong CRF order, excepted in {1, 2}")


def viterbi1o(emits, mask):
    """

    Args:
        emits (torch.Tensor): [batch_size, seq_len, n_labels]
        mask (torch.Tensor): [batch_size, seq_len]

    Returns:

    """
    # emits: [seq_len, batch_size, n_labels]
    emits = emits.transpose(0, 1)
    # mask: [seq_len, batch_size]
    mask = mask.t()

    # phi记录路径
    phi = emits.new_full()


def viterbi2o(emits, mask, label_bos_index):
    """

    Args:
        emits (torch.Tensor): [batch_size, seq_len, n_labels]
        mask (torch.Tensor): [batch_size, seq_len]
        label_bos_index (int):

    Returns:

    """
    batch_size, seq_len, n_labels = emits.shape

    lens = mask.sum(-1)

    # emits: [seq_len, batch_size, n_labels, n_origin_labels, n_origin_labels]
    n_origin_labels = int(n_labels ** 0.5)
    emits = emits.transpose(0, 1)
    emits = emits.view(seq_len, batch_size, n_origin_labels, -1)
    # mask: [seq_len, batch_size]
    mask = mask.t()

    # phi记录路径
    p = emits.new_zeros(seq_len, batch_size, n_origin_labels).long()
    # delta记录分值
    delta = emits.new_zeros(batch_size, n_origin_labels)
    delta += emits[0, :, 0]

    for i in range(1, seq_len):
        # [batch_size, n_labels_pre, 1] + [batch_size, n_labels_pre, n_label_now]
        score = delta.unsqueeze(-1) + emits[i]
        score, _p = score.max(1)
        delta[mask[i]] = score[mask[i]]
        p[i, mask[i]] = _p[mask[i]]
    _, p_end = delta.max(1)

    def backtrack(path, l, now_label):
        labels = [now_label]
        for i in range(l, 0, -1):
            this = path[i][now_label]
            labels.append(this)
            now_label = this
        return list(reversed(labels))

    p = p.permute(1, 0, 2).tolist()
    p_end = p_end.tolist()
    sequences = [backtrack(p[i], length - 1, p_end[i])
                 for i, length in enumerate(lens.tolist())]

    return sequences
