import torch
import torch.nn as nn
import torch.nn.functional as F

class RootWeightPredictorNet(nn.Module):
    def __init__(self):
        super(RootWeightPredictorNet, self).__init__()
        
        # 1. Simplified Encoder / Feature Extractor (Backbone)
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 2. HEAD A: Segmentation Pipeline (Rebuilds the 2.5D image)
        self.dec1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.seg_out = nn.Conv2d(32, 1, kernel_size=1) # Outputs 2.5D Map
        
        # 3. HEAD B: Regression Pipeline (Calculates the scalar weight)
        # Assuming an input image of 256x256 pooled twice -> 64x64 feature map size
        self.flatten_size = 64 * 64 * 64 
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.weight_out = nn.Linear(128, 1) # Outputs exactly 1 scalar float (Grams)

    def forward(self, x):
        # --- Shared Backbone Feature Extraction ---
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        features = F.relu(self.enc2(x))
        bottleneck = self.pool(features) # Deepest understanding of the image
        
        # --- PATHWAY A: 2.5D Segmentation Mapping ---
        # Upsample back to original image size
        up = F.interpolate(bottleneck, scale_factor=4, mode='bilinear', align_corners=True)
        d1 = F.relu(self.dec1(up))
        map_25d = torch.sigmoid(self.seg_out(d1)) # Forces pixels between 0.0 and 1.0
        
        # --- PATHWAY B: Weight Regression ---
        # Flatten the spatial feature grid into a long 1D line of numbers
        flat_features = bottleneck.view(bottleneck.size(0), -1)
        f1 = F.relu(self.fc1(flat_features))
        predicted_weight = self.weight_out(f1) # Raw scalar regression value
        
        return map_25d, predicted_weight
