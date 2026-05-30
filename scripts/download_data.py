import subprocess
import zipfile
from pathlib import Path
import sys

data_dir = Path("data")

data_dir.mkdir(parents=True, exist_ok=True)

def unzip_files(data_dir: Path):
    """Extract downloaded zip files."""
    for zip_file in data_dir.glob("*.zip"):
        print(f"Extracting {zip_file.name}")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(data_dir)

        zip_file.unlink()


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    unzip_files(raw_dir)

    print("Contents:", list(raw_dir.iterdir()))

    print("Dataset ready in:", raw_dir.resolve())


if __name__ == "__main__":
    main()
