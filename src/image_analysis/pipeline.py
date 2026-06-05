from pathlib import Path

from image_analysis.binarization import apply_advanced_binarizations
from image_analysis.evaluate_masks import evaluate_binarization_methods
from image_analysis.batch_processing import process_entire_dataset

RAW_DIR = Path("data/raw")
GT_DIR = Path("data/ground_truth_masks")
TARGET_DIR = Path("data/targets")

def main():
    image_path = RAW_DIR / "example.png"
    gt_path = GT_DIR / "example.png"

    adaptive, frangi, morph = apply_advanced_binarizations(
        str(image_path),
        output_prefix="outputs/example"
    )

    results = evaluate_binarization_methods(
        adaptive,
        frangi,
        morph,
        ensemble,
        str(gt_path)
    )

    winner = results["winner"].lower()

    if winner == "morphological":
        winner = "morphological"
    elif winner == "frangi":
        winner = "frangi"
    elif winner == "adaptive"
        winner = "adaptive"
    else:
        winner = "ensemble"

    process_entire_dataset(
        input_folder=str(RAW_DIR),
        output_folder=str(TARGET_DIR),
        winning_method=winner
    )

if __name__ == "__main__":
    main()
