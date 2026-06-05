import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Allows imports from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from image_analysis.weight_function import RootWeightPredictorNet
from image_analysis.losses.huberloss_function import DenseThinRootHuberLoss

import argparse
import random
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/demo")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RootWeightDataset(Dataset):
    def __init__(self, csv_path, image_dir, target_dir=None, image_size=256):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.target_dir = Path(target_dir) if target_dir else None

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_dir / row["image_filename"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        weight = torch.tensor(row["root_weight"], dtype=torch.float32)

        if self.target_dir:
            target_path = self.target_dir / row["image_filename"].replace(".png", "_target.npy")
            target_map = torch.tensor(
                __import__("numpy").load(target_path),
                dtype=torch.float32
            ).unsqueeze(0)
            return image, target_map, weight

        return image, weight


def train():
    csv_path = "data/root_weights.csv"
    image_dir = "data/raw"
    target_dir = "data/targets"  

    batch_size = 8
    epochs = 50
    lr = 1e-4
    aux_map_weight = 0.05

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = RootWeightDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        target_dir=target_dir,
        image_size=256,
    )

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = RootWeightPredictorNet().to(device)

    weight_loss_fn = nn.SmoothL1Loss()
    map_loss_fn = DenseThinRootHuberLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, target_maps, weights in train_loader:
            images = images.to(device)
            target_maps = target_maps.to(device)
            weights = weights.to(device)

            pred_maps, pred_weights = model(images)

            pred_weights = pred_weights.squeeze()

            weight_loss = weight_loss_fn(pred_weights, weights)
            map_loss = map_loss_fn(pred_maps, target_maps)

            loss = weight_loss + aux_map_weight * map_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for images, target_maps, weights in val_loader:
                images = images.to(device)
                target_maps = target_maps.to(device)
                weights = weights.to(device)

                pred_maps, pred_weights = model(images)
                pred_weights = pred_weights.squeeze()

                weight_loss = weight_loss_fn(pred_weights, weights)
                map_loss = map_loss_fn(pred_maps, target_maps)

                loss = weight_loss + aux_map_weight * map_loss

                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(pred_weights - weights)).item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_root_weight_model.pt")
            print("Saved best model")

    print("Training complete.")

def main():
    args = parse_args()
    set_seed(args.seed)
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
