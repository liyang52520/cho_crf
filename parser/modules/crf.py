import torch
import torch.nn as nn


class CRF(nn.Module):
    """

    Args:
        order (int):
            CRF阶数，1 or 2
        bos_index (int):
            当阶数为2时，是需要<bos>标签为起始计算标签的
        pad_index:
            在使用viterbi时会用到
    """

    def __init__(self, order, n_labels, bos_index, pad_index):
        super(CRF, self).__init__()
        self.order = order
        assert self.order in {1, 2}, "order must be 1 or 2."
        self.n_labels = n_labels
        self.bos_index = bos_index
        self.pad_index = pad_index

    def extra_repr(self):
        s = f"order={self.order}"
        return s

    def forward(self, emits, targets, mask):
        """
        返回CRF层loss

        Args:
            emits (torch.Tensor): [batch_size, seq_len, n_labels ** order]
            targets (torch.Tensor): [batch_size, seq_len + order - 1]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        total_token = mask.sum().float()

        # 改变emits，target和mask的维度
        # emits: [seq_len, batch_size, n_labels ** order]
        emits = emits.transpose(0, 1)
        # targets: [seq_len + order - 1, batch_size]
        targets = targets.t()
        # mask: [seq_len, batch_size]
        mask = mask.t()

        # log_z
        log_z = self.compute_log_z(emits, mask)

        if self.order == 2:
            # targets: [seq_len, batch_size]
            targets = targets[:-1] * self.n_labels + targets[1:]

        # scores: gather([seq_len, batch_size, n_labels ** order], -1, [seq_len, batch_size, 1]).squeeze(-1)
        #         => [seq_len, batch_size] => [n]
        scores = emits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).masked_select(mask).sum()

        # loss
        loss = (log_z - scores) / total_token

        return loss

    def compute_log_z(self, emits, mask):
        seq_len, batch_size = mask.shape
        n_labels = self.n_labels

        # 因为double精度更高吗?
        emits = emits.double()

        # log_alpha: [batch_size, n_label]
        log_alpha = emits.new_zeros(batch_size, n_labels, dtype=torch.double)

        # 计算log_alpha的第一个状态
        if self.order == 1:
            # 初始为发射到第一个词的分值（uni_gram不需要转移分值）
            # emits: [seq_len, batch_size, n_labels]
            log_alpha += emits[0, :]
        elif self.order == 2:
            # emits: [seq_len, batch_size, n_labels, n_labels]
            emits = emits.view(seq_len, batch_size, n_labels, -1)
            log_alpha += emits[0, :, self.bos_index, :]
        else:
            raise Exception("wrong CRF order, excepted in {1, 2}")

        # 计算剩余状态
        for i in range(1, seq_len):
            # 对于1阶
            # scores: [batch_size, n_labels, 1] + [batch_size, 1, n_labels]=> [batch_size, n_labels, n_labels]
            if self.order == 1:
                scores = log_alpha.unsqueeze(-1) + emits[i].unsqueeze(1)
            # 对于2阶：
            # scores: [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels]
            elif self.order == 2:
                scores = log_alpha.unsqueeze(-1) + emits[i]
            else:
                raise Exception("wrong CRF order, excepted in {1, 2}")

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
            emits (torch.Tensor): [batch_size, seq_len, n_labels ** order]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        if self.order == 2:
            emits = emits.view(*mask.shape, self.n_labels, -1)
        emits = emits.transpose(0, 1)
        mask = mask.t()
        return self.viterbi_1o_2o(emits, mask)

    def viterbi_1o_2o(self, emits, mask):
        """
        1阶或2阶Viterbi解码，两者大体是一致的
        Args:
            emits (torch.Tensor): [seq_len, batch_size, n_labels] or [batch_size, seq_len, n_labels, n_labels]
            mask (torch.Tensor): [seq_len, batch_size]

        Returns:

        """
        seq_len, batch_size = mask.shape
        n_labels = self.n_labels
        last_next_position = mask.sum(0)

        # phi记录路径
        # phi: [seq_len + 1, batch_size, n_labels]，初始化全部指向<pad>
        phi = torch.full((seq_len + 1, batch_size, n_labels), self.pad_index, dtype=torch.long).to(emits.device)

        # delta记录分值
        # delta: [batch_size, n_labels]
        if self.order == 1:
            delta = emits[0].clone()
        else:
            # 初始为<bos>到其他tag
            delta = emits[0, :, self.bos_index, :].clone()

        # 计算后续状态
        for i in range(1, seq_len):

            if self.order == 1:
                # score: [batch_size, n_labels, 1] + [batch_size, 1, n_labels] => [batch_size, n_labels, n_labels]
                score = delta.unsqueeze(-1) + emits[i].unsqueeze(1)
            else:
                # score: [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels]
                #       => [batch_size, n_labels, n_labels]
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
