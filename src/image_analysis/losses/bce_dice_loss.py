import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice Loss.

    Args:
        bce_weight (float): Weight applied to BCE loss.
        dice_weight (float): Weight applied to Dice loss.
        smooth (float): Smoothing factor to avoid division by zero.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def dice_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        dice_score = (
            (2.0 * intersection + self.smooth)
            / (predictions.sum() + targets.sum() + self.smooth)
        )

        return 1.0 - dice_score

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            predictions,
            targets.float(),
        )

        dice = self.dice_loss(predictions, targets)

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )
