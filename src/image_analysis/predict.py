from pathlib import Path

import cv2
import numpy as np
import torch

from image_analysis.train import TinyUNet


MODEL_PATH = Path("models/root_distance_model.pth")
INPUT_DIR = Path("data/raw")
PREDICTION_DIR = Path("outputs/predictions")


def predict():
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    image_paths = [
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    ]

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image_norm = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(tensor)

        pred_map = prediction.squeeze().cpu().numpy()

        npy_path = PREDICTION_DIR / f"{image_path.stem}_prediction.npy"
        png_path = PREDICTION_DIR / f"{image_path.stem}_prediction.png"

        np.save(npy_path, pred_map)

        pred_img = (pred_map * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(png_path), pred_img)

        print(f"Saved prediction: {png_path}")

    print("Prediction complete.")


if __name__ == "__main__":
    predict()
