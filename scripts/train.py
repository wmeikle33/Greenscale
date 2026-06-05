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

args = parse_args()
set_seed(args.seed)
