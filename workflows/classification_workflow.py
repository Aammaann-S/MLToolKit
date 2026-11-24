"""Train and score logistic regression on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MLToolKit.toolkit_loader import load_impl


def main():
    data = load_impl("implementations/09_Utilities/synthetic_datasets.py")
    trainer = load_impl("implementations/04_Classification/logistic_regression.py")
    frame = data.classification_frame()
    result = trainer.train_logistic_regression(frame.drop(columns=["target"]), frame["target"])
    accuracy = result["metrics"]["accuracy"]
    print("classification_accuracy", round(accuracy, 4))
    if accuracy < 0.7:
        raise SystemExit("Classification workflow accuracy was below 0.7")
    return result


if __name__ == "__main__":
    main()
