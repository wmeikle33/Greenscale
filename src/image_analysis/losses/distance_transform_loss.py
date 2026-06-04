import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn


class DistanceTransformLoss(nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: raw model outputs, shape [B, 1, H, W]
        targets: binary masks, shape [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        distance_maps = []

        for target in targets.detach().cpu().numpy():
            mask = target[0] > 0.5

            # Distance from every background pixel to nearest root pixel
            distance = distance_transform_edt(~mask)

            if self.normalize and distance.max() > 0:
                distance = distance / distance.max()

            distance_maps.append(distance)

        distance_maps = torch.tensor(
            np.stack(distance_maps),
            dtype=probs.dtype,
            device=probs.device,
        ).unsqueeze(1)

        false_positive_penalty = probs * distance_maps
        false_negative_penalty = (1.0 - probs) * targets

        loss = false_positive_penalty.mean() + false_negative_penalty.mean()

        return loss
