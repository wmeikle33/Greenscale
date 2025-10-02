import argparse, json
from .adapters_npz import NPZLoader
from .preprocess import Normalize01
from .models import CentroidClassifier
from .pipeline import run_train_eval

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    metrics = run_train_eval(
        NPZLoader(args.train), NPZLoader(args.test),
        Normalize01(flatten=True), CentroidClassifier()
    )
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
