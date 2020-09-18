import torch
import torch.nn as nn


class CRF(nn.Module):
    """

    Args:
        bos_index (int):
            当阶数为2时，是需要<bos>标签为起始计算标签的
        pad_index:
            在使用viterbi时会用到
    """

    def __init__(self, n_labels, bos_index, pad_index):
        super(CRF, self).__init__()
        self.n_labels = n_labels
        self.bos_index = bos_index
        self.pad_index = pad_index

    def extra_repr(self):
        s = f"n_labels={self.n_labels}"
        return s

    def forward(self, emits, targets, mask):
        """
        返回CRF层loss

        Args:
            emits (torch.Tensor): [batch_size, seq_len, n_labels, n_labels]
            targets (torch.Tensor): [batch_size, seq_len + 1]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        total_token = mask.sum().float()

        # 改变emits，target和mask的维度
        # emits: [seq_len, batch_size, n_labels, n_labels]
        emits = emits.transpose(0, 1)
        # targets: [seq_len + 1, batch_size]
        targets = targets.t()
        # mask: [seq_len, batch_size]
        mask = mask.t()

        # log_z
        log_z = self.compute_log_z(emits, mask)

        # targets: [seq_len, batch_size]
        targets = targets[:-1] * self.n_labels + targets[1:]

        # scores: gather([seq_len, batch_size, n_labels ** 2], -1, [seq_len, batch_size, 1]).squeeze(-1)
        #         => [seq_len, batch_size] => [n]
        emits = emits.view(*mask.shape, -1)
        scores = emits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).masked_select(mask).sum()

        # loss
        loss = (log_z - scores) / total_token

        return loss

    def compute_log_z(self, emits, mask):
        """

        Args:
            emits: [seq_len, batch_size, n_labels, n_labels]
            mask: [seq_len, batch_size]

        Returns:

        """
        seq_len, batch_size = mask.shape
        n_labels = self.n_labels

        # 因为double精度更高吗?
        emits = emits.double()

        # log_alpha: [batch_size, n_label]
        log_alpha = emits.new_zeros(batch_size, n_labels, dtype=torch.double)
        # 计算log_alpha的第一个状态
        log_alpha += emits[0, :, self.bos_index, :]

        # 计算剩余状态
        for i in range(1, seq_len):
            # scores: [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels]
            scores = log_alpha.unsqueeze(-1) + emits[i]

            # temp_log_alpha: [batch_size, n_labels]
            temp_log_alpha = torch.logsumexp(scores, dim=1)

            # 根据mask更新log_alpha
            log_alpha[mask[i]] = temp_log_alpha[mask[i]]

        # log_z: [batch_size]
        log_z = torch.logsumexp(log_alpha, dim=1)

        return log_z.sum()

    def viterbi(self, emits, mask):
        """

        Args:
            emits (torch.Tensor): [batch_size, seq_len, n_labels, n_labels]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        batch_size, seq_len = mask.shape
        n_labels = self.n_labels
        last_next_position = mask.sum(1)

        # emits: [seq_len, batch_size, n_labels, n_labels]
        emits = emits.transpose(0, 1)
        # mask: [seq_len, batch_size]
        mask = mask.t()

        # phi记录路径
        # phi: [seq_len + 1, batch_size, n_labels]，初始化全部指向<pad>
        phi = torch.full((seq_len + 1, batch_size, n_labels), self.pad_index, dtype=torch.long).to(emits.device)

        # delta记录分值
        # delta: [batch_size, n_labels]，初始为<bos>到其他tag
        delta = emits[0, :, self.bos_index, :].clone()

        # 计算后续状态
        for i in range(1, seq_len):
            # score: [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels] => [batch_size, n_labels, n_labels]
            score = delta.unsqueeze(-1) + emits[i]
            # temp_delta, phi[i]: [batch_size, n_labels]
            temp_delta, phi[i] = torch.max(score, dim=1)

            # 根据mask决定是否调整phi和delta
            delta[mask[i]] = temp_delta[mask[i]]
            phi[i][~mask[i]] = self.pad_index

        # 将每一个句子有效末尾后面一个位置的<pad>位指向的tag改为delta中记录的最大tag
        batch = torch.arange(batch_size, dtype=torch.long).to(emits.device)
        phi[last_next_position, batch, self.pad_index] = torch.argmax(delta, dim=-1)

        # tags: [seq_len, batch_size]
        tags = torch.zeros((seq_len, batch_size), dtype=torch.long).to(emits.device)
        # pre_tags: [batch_size, 1]
        pre_tags = torch.full((batch_size, 1), self.pad_index, dtype=torch.long).to(emits.device)
        for i in range(seq_len, 0, -1):
            j = i - seq_len - 1
            # pre_tags: [batch_size, 1]
            pre_tags = torch.gather(phi[i], 1, pre_tags)
            tags[j] = pre_tags.squeeze()
        # tags: [batch_size, seq_len]
        tags = tags.t()
        return tags
