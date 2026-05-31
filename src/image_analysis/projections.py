import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.signal import convolve2d

def distance_layer(mask):
    return distance_transform_edt(mask > 0)
    
def skeleton_layer(mask):
    return skeletonize(mask > 0).astype(np.float32)
  
def depth_density(mask):
    density = mask.mean(axis=1)
    return np.repeat(
        density[:, None],
        mask.shape[1],
        axis=1
    )
    
def width_density(mask):
    density = mask.mean(axis=0)
    return np.repeat(
        density[None, :],
        mask.shape[0],
        axis=0
    )

def branch_layer(mask):
    skel = skeletonize(mask > 0).astype(np.uint8)

    kernel = np.ones((3,3))
    neighbors = convolve2d(
        skel,
        kernel,
        mode="same"
    )

    return (neighbors >= 4).astype(np.float32)

def component_layer(mask):
    return label(mask > 0).astype(np.float32)

def build_25d(mask):

    return np.stack([
        mask.astype(np.float32),
        distance_layer(mask),
        skeleton_layer(mask),
        depth_density(mask),
        width_density(mask),
        branch_layer(mask),
    ], axis=-1)

def save_25d(mask: np.ndarray, output_path: str | Path) -> Path:
    """Create and save a 2.5D `.npy` file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = build_25d(mask)
    np.save(output_path, features)

    return output_path
