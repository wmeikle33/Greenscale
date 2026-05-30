import argparse
from pathlib import Path

from src.create_masks import mouse_callback_paint
from src.create_masks import automated_generation_with_manual_edit
from .data import load_csv
from .model import train_eval_save
image_path, sam_checkpoint_path, output_dir

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to training CSV",
    )
    ap.add_argument("--image_path", type=str)
    ap.add_argument("--sam_checkpoint_path", type=str)
    ap.add_argument("--output_dir", type=str)
    return ap.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model_path)

    df = load_csv(csv_path, nrows=args.nrows)


if __name__ == "__main__":
    main()
