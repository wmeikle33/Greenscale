import numpy as np 
import cv2
from skimage.filters import frangi
from skimage.filters import threshold_otsu


def apply_advanced_binarizations(image_path, output_prefix="root_mask"):
    """
    Applies three different binarization techniques to an input image
    and returns them as strict binary arrays (0 and 1).
    """
    # 1. Load the original image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # -------------------------------------------------------------------------
    # METHOD 1: Local Adaptive Thresholding
    # -------------------------------------------------------------------------
    adaptive_raw = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    # Convert from 0/255 to strict deep learning target 0/1
    mask_adaptive = (adaptive_raw > 127).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_adaptive = cv2.morphologyEx(mask_adaptive, cv2.MORPH_OPEN, kernel)
    mask_adaptive = cv2.morphologyEx(mask_adaptive, cv2.MORPH_CLOSE, kernel)

    # -------------------------------------------------------------------------
    # METHOD 2: Hessian / Frangi Ridge Filtering
    # -------------------------------------------------------------------------
    img_float = img.astype(np.float32) / 255.0
    ridge_probability = frangi(img_float, sigmas=np.arange(1, 4, 1))
    t = threshold_otsu(ridge_probability)
    mask_frangi = (ridge_probability > t)
    kernel = np.ones((3,3), np.uint8)
    mask_frangi = cv2.morphologyEx(mask_frangi, cv2.MORPH_OPEN, kernel)
    mask_frangi = cv2.morphologyEx(mask_frangi, cv2.MORPH_CLOSE, kernel)
    
    # -------------------------------------------------------------------------
    # METHOD 3: Morphological Contrast Enhancement (Top-Hat)
    # -------------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Top-hat isolates features brighter than their surroundings within the 3x3 region
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    # Boost the contrast of fine details heavily
    enhanced_img = cv2.addWeighted(img, 1.0, tophat, 2.0, 0)
    
    # Apply a standard threshold to the freshly amplified thin structures
    _, enhanced_raw = cv2.threshold(enhanced_img, 70, 255, cv2.THRESH_BINARY)
    mask_morph = (enhanced_raw > 127).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)

    # -------------------------------------------------------------------------
    # METHOD 4: Combined Method
    # -------------------------------------------------------------------------

    combined = (
    mask_adaptive.astype(int)
    + mask_frangi.astype(int)
    + mask_morph.astype(int)
    )
    mask_ensemble = (combined >= 2).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Optional: Save the masks back to disk to inspect visually (as 0-255 images)
    # -------------------------------------------------------------------------
    cv2.imwrite(f"{output_prefix}_adaptive.png", mask_adaptive * 255)
    cv2.imwrite(f"{output_prefix}_frangi.png", mask_frangi * 255)
    cv2.imwrite(f"{output_prefix}_morphological.png", mask_morph * 255)
    cv2.imwrite(f"{output_prefix}_combined.png", mask_combined * 255)

    return mask_adaptive, mask_frangi, mask_morph, mask_combined
