import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseThinRootHuberLoss(nn.Module):
    def __init__(self, delta=0.1, beta_foreground=25.0, beta_separation=10.0):
        """
        Custom Spatially-Adaptive Loss for Dense, Highly-Branched Thin Roots.
        
        Parameters:
        - delta: Huber threshold (optimized for thin 0.0 - 1.0 fractional gradients).
        - beta_foreground: Scale factor to balance tiny thin roots vs vast soil backgrounds.
        - beta_separation: Massive penalty factor for failing to split closely packed branches.
        """
        super(DenseThinRootHuberLoss, self).__init__()
        self.delta = delta
        self.beta_fg = beta_foreground
        self.beta_sep = beta_separation
        self.base_huber = nn.HuberLoss(delta=delta, reduction='none')

    def forward(self, predictions, targets):
        # 1. Compute the raw Huber loss for every pixel
        raw_huber_loss = self.base_huber(predictions, targets)
        
        # 2. Extract basic foreground root mask (Target value > 0)
        fg_mask = (targets > 0.0).float()
        
        # 3. COMPUTE THE SEPARATION MAP (Find close-proximity valleys)
        # We use a max-pooling operation to create a structural "halo" around the tracks.
        # This highlights background pixels trapped right between nearby thin lines.
        dilated_targets = F.max_pool2d(targets, kernel_size=5, stride=1, padding=2)
        
        # Separation boundaries exist where there is a dilated neighbor presence,
        # but the actual ground truth pixel is pure background soil (0.0)
        separation_valleys = (dilated_targets > 0.0).float() - fg_mask
        separation_valleys = torch.clamp(separation_valleys, min=0.0, max=1.0)
        
        # 4. Construct the Spatially-Adaptive Weight Matrix (Omega)
        # - Soil Pixels get a baseline weight of 1.0
        # - Thin root tracks get boosted by beta_fg
        # - Critical empty valleys between branches get boosted heavily by beta_sep
        weight_matrix = 1.0 + (self.beta_fg - 1.0) * fg_mask + self.beta_sep 
        * separation_valleys
        
        # 5. Apply the spatial weights and reduce to batch mean scalar
        weighted_loss = weight_matrix * raw_huber_loss
        return torch.mean(weighted_loss)
