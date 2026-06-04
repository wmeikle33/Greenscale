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

    cv2.imwrite(str(RAW_DIR / "example.png"), image)
    cv2.imwrite(str(MASK_DIR / "example.png"), mask)

    target = (mask / 255.0).astype("float32")
    np.save(TARGET_DIR / "example_target.npy", target)

    root_weights = pd.DataFrame(
        {
            "image_filename": ["example.png"],
            "root_weight": [1.2],
        }
    )
    root_weights.to_csv(DATA_DIR / "root_weights.csv", index=False)

    print(f"Wrote demo image to {RAW_DIR}")
    print(f"Wrote demo mask to {MASK_DIR}")
    print("Wrote data/root_weights.csv")


if __name__ == "__main__":
    main()
