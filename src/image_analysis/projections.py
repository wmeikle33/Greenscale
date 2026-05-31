import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def distance_layer(mask):
    return distance_transform_edt(mask > 0)
  
def depth_density(mask):
    density = mask.mean(axis=1)
    return np.repeat(
        density[:, None],
        mask.shape[1],
        axis=1
    )
