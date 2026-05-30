import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import frangi

def process_entire_dataset(input_folder, output_folder, winning_method="frangi"):
    """
    Loops through an entire folder of raw root images, binarizes them 
    using the winning method, calculates the normalized distance transform,
    and saves the final deep learning target maps.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Get all images in the input directory
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Starting batch processing for {len(image_files)} images...")
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # --- STEP A: Apply the mathematically chosen binarization ---
        if winning_method == "adaptive":
            raw_mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            binary_mask = (raw_mask > 127).astype(np.uint8)
        elif winning_method == "frangi":
            prob = frangi(img, sigmas=np.arange(1, 4, 1))
            binary_mask = (prob > 0.05).astype(np.uint8)
        elif winning_method == "morphological":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            enhanced = cv2.addWeighted(img, 1.0, tophat, 2.0, 0)
            _, raw_mask = cv2.threshold(enhanced, 70, 255, cv2.THRESH_BINARY)
            binary_mask = (raw_mask > 127).astype(np.uint8)
            
        # --- STEP B: Generate the Distance Transform (Back to your original goal!) ---
        # Using standard EDT as an example; swap with Truncated or Gaussian if needed
        distance_map = ndimage.distance_transform_edt(binary_mask)
        
        if distance_map.max() > 0:
            distance_map_normalized = distance_map / distance_map.max()
        else:
            distance_map_normalized = distance_map
            
        # --- STEP C: Save the final float32 target map ---
        # We save as a .npy (NumPy binary file) to preserve the exact floating-point decimals 
        # between 0.0 and 1.0 without image compression ruining the math.
        output_filename = os.path.splitext(filename)[0] + "_target.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, distance_map_normalized.astype(np.float32))

    print("Batch processing complete! All target maps saved successfully.")

# Example execution:
# process_entire_dataset("path/to/raw_images", "path/to/dl_targets", winni
