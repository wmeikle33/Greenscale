import argparse
from pathlib import Path

from .data import load_csv
from .model import train_eval_save

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.gz"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to training CSV",
    )
    ap.add_argument("--label", default="click", help="Target column")
    ap.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Saved model path",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--nrows", type=int, default=200000, help="Rows to load for training")
    ap.add_argument(
        "--model",
        choices=["logreg", "xgb"],
        default="logreg",
        help="Which model to train.",
    )
    ap.add_argument("--search", action="store_true", help="Enable hyperparameter search")
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

    metrics = train_eval_save(
        df=df,
        label=args.label,
        model_path=model_path,
        model_name=args.model,
        random_state=args.random_state,
        test_size=args.test_size,
        param_search=args.search,
        n_iter=args.n_iter,
        cv=args.cv,
    )

    print(f"Saved model to: {model_path}")
    print(f"log_loss={metrics['log_loss']:.6f}")
    if "auc" in metrics:
        print(f"auc={metrics['auc']:.6f}")


if __name__ == "__main__":
    main()
