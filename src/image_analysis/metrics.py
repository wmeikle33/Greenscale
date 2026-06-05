def calculate_metrics(pred_mask, true_mask):
    tp = ((pred_mask == 1) & (true_mask == 1)).sum()
    fp = ((pred_mask == 1) & (true_mask == 0)).sum()
    fn = ((pred_mask == 0) & (true_mask == 1)).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }
