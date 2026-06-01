
from pathlib import Path

from image_analysis.binarization import apply_advanced_binarizations
from image_analysis.evaluate_masks import evaluate_binarization_methods
from image_analysis.batch_processing import process_entire_dataset

RAW_DIR = Path("data/raw")
GT_DIR = Path("data/ground_truth_masks")
TARGET_DIR = Path("data/targets")

def main():
    # 1. Pick one representative image + ground truth mask
    image_path = RAW_DIR / "example.png"
    gt_path = GT_DIR / "example.png"

    # 2. Generate candidate masks
    adaptive, frangi, morph = apply_advanced_binarizations(
        str(image_path),
        output_prefix="outputs/example"
    )

    # 3. Evaluate methods
    scores = evaluate_binarization_methods(
        adaptive,
        frangi,
        morph,
        str(gt_path)
    )

    # 4. Pick winner
    winner = max(scores, key=scores.get).lower()

    # normalize name expected by batch_processing.py
    if winner == "morphological":
        winner = "morphological"
    elif winner == "frangi":
        winner = "frangi"
    else:
        winner = "adaptive"

    # 5. Process all images into training targets
    process_entire_dataset(
        input_folder=str(RAW_DIR),
        output_folder=str(TARGET_DIR),
        winning_method=winner
    )

if __name__ == "__main__":
    main()
