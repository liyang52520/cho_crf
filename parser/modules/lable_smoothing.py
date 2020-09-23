import torch.nn as nn


class LabelSmoothing(nn.Module):
    """
    label smoothing，给emits平滑一下分值用看看效果如何
    """

    def __init__(self, n_labels, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.n_labels = n_labels
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): [batch_size, seq_len, n_labels]

        Returns:

        """
        return (1 - self.smoothing) * x + self.smoothing / self.n_labels
