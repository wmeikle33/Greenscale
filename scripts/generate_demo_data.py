from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MASK_DIR = ROOT / "data" / "ground_truth_masks"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    image = np.full((256, 256), 235, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)

    cv2.circle(image, (90, 120), 45, 80, -1)
    cv2.rectangle(image, (145, 70), (210, 170), 120, -1)

    cv2.circle(mask, (90, 120), 45, 255, -1)
    cv2.rectangle(mask, (145, 70), (210, 170), 255, -1)

    cv2.imwrite(str(RAW_DIR / "demo_001.png"), image)
    cv2.imwrite(str(MASK_DIR / "demo_001_mask.png"), mask)

    print(f"Wrote demo image to {RAW_DIR}")
    print(f"Wrote demo mask to {MASK_DIR}")


if __name__ == "__main__":
    main()
