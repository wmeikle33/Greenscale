from pathlib import Path

from image_analysis.batch_processing import process_entire_dataset
from image_analysis.binarization import apply_advanced_binarizations
from image_analysis.evaluate_masks import evaluate_binarization_methods


def run_pipeline(data_dir: Path, output_dir: Path) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    raw_dir = data_dir / "raw"
    gt_dir = data_dir / "ground_truth_masks"
    target_dir = output_dir / "targets"

    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(raw_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {raw_dir}")

    image_path = image_paths[0]
    gt_path = gt_dir / image_path.name

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {gt_path}")

    output_prefix = output_dir / image_path.stem

    adaptive, frangi, morph, ensemble = apply_advanced_binarizations(
        str(image_path),
        output_prefix=str(output_prefix),
    )

    results = evaluate_binarization_methods(
        adaptive,
        frangi,
        morph,
        ensemble,
        str(gt_path),
    )

    winner = results["winner"].lower()

    if winner not in {"morphological", "frangi", "adaptive", "ensemble"}:
        winner = "ensemble"

    process_entire_dataset(
        input_folder=str(raw_dir),
        output_folder=str(target_dir),
        winning_method=winner,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    run_pipeline(
        data_dir=root / "data",
        output_dir=root / "outputs",
    )


if __name__ == "__main__":
    main()
