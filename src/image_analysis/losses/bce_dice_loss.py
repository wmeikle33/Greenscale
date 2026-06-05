import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight=0.3,
        dice_weight=0.7,
        pos_weight=None,
        smooth=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, predictions, targets):
        targets = targets.float()

        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor(
                [self.pos_weight],
                device=predictions.device,
                dtype=predictions.dtype,
            )

        bce = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=pos_weight,
        )

        probs = torch.sigmoid(predictions)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = 1 - (
            (2 * intersection + self.smooth)
            / (probs.sum() + targets.sum() + self.smooth)
        )

        return self.bce_weight * bce + self.dice_weight * dice
