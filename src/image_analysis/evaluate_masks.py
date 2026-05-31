import numpy as np
import cv2

def calculate_segmentation_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(bool)
    gt = (gt_mask > 0).astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union
        
    # Calculate Dice Coefficient (F1-Score)
    total_pixels_predicted_and_gt = pred.sum() + gt.sum()
    if total_pixels_predicted_and_gt == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = (2.0 * intersection) / total_pixels_predicted_and_gt
        
    return iou, dice

def evaluate_binarization_methods(mask_adaptive, mask_frangi, mask_morph, ground_truth_path):
    gt_img = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise FileNotFoundError(f"Missing Ground Truth file at: {ground_truth_path}")
    gt_mask = (gt_img > 127).astype(np.uint8)
    
    iou_adapt, dice_adapt = calculate_segmentation_metrics(mask_adaptive, gt_mask)
    iou_fran,  dice_fran  = calculate_segmentation_metrics(mask_frangi,   gt_mask)
    iou_morph, dice_morph = calculate_segmentation_metrics(mask_morph,    gt_mask)
    
    print("\n================ BINARIZATION METRICS SCOREBOARD ================")
    print(f"Method 1: Adaptive Thresholding       | IoU: {iou_adapt:.4f} | Dice: {dice_adapt:.4f}")
    print(f"Method 2: Frangi Ridge Filter        | IoU: {iou_fran:.4f}  | Dice: {dice_fran:.4f}")
    print(f"Method 3: Morphological Top-Hat       | IoU: {iou_morph:.4f} | Dice: {dice_morph:.4f}")
    print("=================================================================\n")
    
    scores = {"Adaptive": iou_adapt, "Frangi": iou_fran, "Morphological": iou_morph}
    winner = max(scores, key=scores.get)
    
    return winner
