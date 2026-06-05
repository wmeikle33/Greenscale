from image_analysis.pipeline import main as run_pipeline
from image_analysis.train import main as run_train


def main() -> None:
    run_pipeline()
    run_train()


if __name__ == "__main__":
    main()
