import torch
from torch import nn


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        error = torch.abs(target - pred)

        alpha_minus_y = self.alpha - target

        small_error_loss = (
            self.omega
            * torch.log(
                1
                + torch.pow(error / self.epsilon, alpha_minus_y)
            )
        )

        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, alpha_minus_y)))
            * alpha_minus_y
            * torch.pow(self.theta / self.epsilon, alpha_minus_y - 1)
            / self.epsilon
        )

        C = (
            self.theta * A
            - self.omega
            * torch.log(
                1
                + torch.pow(self.theta / self.epsilon, alpha_minus_y)
            )
        )

        large_error_loss = A * error - C

        loss = torch.where(error < self.theta, small_error_loss, large_error_loss)

        return loss.mean()
