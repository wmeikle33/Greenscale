import argparse
from pathlib import Path

from src.create_masks import mouse_callback_paint
from src.create_masks import mouse_callback_paint
from .data import load_csv
from .model import train_eval_save


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to training CSV",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=20, help="Number of parameter samples")
    ap.add_argument("--cv", type=int, default=3, help="CV folds for search")
    return ap.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model_path)

    df = load_csv(csv_path, nrows=args.nrows)

    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in {csv_path}")


    print(f"Saved model to: {model_path}")
    print(f"log_loss={metrics['log_loss']:.6f}")
    if "auc" in metrics:
        print(f"auc={metrics['auc']:.6f}")


if __name__ == "__main__":
    main()
