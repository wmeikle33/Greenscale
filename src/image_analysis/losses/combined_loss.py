import torch
import torch.nn.functional as F
from torch import nn


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    skeleton = F.relu(x - soft_open(x))

    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations: int = 10, smooth: float = 1e-6):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: raw model outputs, shape [B, 1, H, W]
        targets: binary masks, shape [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        pred_skeleton = soft_skeletonize(probs, self.iterations)
        target_skeleton = soft_skeletonize(targets, self.iterations)

        tprec = ((pred_skeleton * targets).sum(dim=(1, 2, 3)) + self.smooth) / (
            pred_skeleton.sum(dim=(1, 2, 3)) + self.smooth
        )

        tsens = ((target_skeleton * probs).sum(dim=(1, 2, 3)) + self.smooth) / (
            target_skeleton.sum(dim=(1, 2, 3)) + self.smooth
        )

        cl_dice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)

        return 1.0 - cl_dice.mean()
