import torch
from torch import nn


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.5,
        smooth: float = 1e-6,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()

        fp = ((1 - targets) * probs).sum()

        fn = (targets * (1 - probs)).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = torch.pow((1.0 - tversky), self.gamma)

        return loss
