import torch


class SequenceLoss(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, y_pred, y):
        if len(y_pred.shape) > 2:
            batch_size, max_seq_len = y_pred.shape[:2]
            y_pred = y_pred.view(batch_size * max_seq_len, -1)
            y = y.view(batch_size * max_seq_len)
        return self.loss(y_pred, y)
