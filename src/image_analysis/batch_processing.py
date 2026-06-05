import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import frangi

def process_entire_dataset(input_folder, output_folder, winning_method):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder) if 
    f.lower().endswith(valid_extensions)]
    
    print(f"Starting batch processing for {len(image_files)} images...")
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        if winning_method == "adaptive":
            raw_mask = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
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

        else:
            raise ValueError(f"Unknown winning_method: {winning_method}")

        distance_map = ndimage.distance_transform_edt(binary_mask)

        if distance_map.max() > 0:
            distance_map_normalized = distance_map / distance_map.max()
        else:
            distance_map_normalized = distance_map

        output_filename = os.path.splitext(filename)[0] + "_target.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, distance_map_normalized.astype(np.float32))
