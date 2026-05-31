import argparse
from pathlib import Path

from src.batch_processing import process_entire_dataset

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
    
    image = Path(args.image_path).expanduser().resolve()
    sam_checkpoint = Path(args.sam_checkpoint_path)
    ouput = Path(args.output_dir)
    process_entire_dataset(image, sam_checkpoint, output)


if __name__ == "__main__":
    main()
