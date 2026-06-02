from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth_masks"
TARGET_DIR = DATA_DIR / "targets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
