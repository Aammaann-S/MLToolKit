"""Train and score a linear regression model on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MLToolKit.toolkit_loader import load_impl


def main():
    data = load_impl("implementations/09_Utilities/synthetic_datasets.py")
    trainer = load_impl("implementations/03_Regression/multiple_linear_regression.py")
    frame = data.regression_frame()
    result = trainer.train_multiple_linear_regression(frame.drop(columns=["target"]), frame["target"])
    print("regression_r2", round(result["metrics"]["r2"], 4))
    if result["metrics"]["r2"] < 0.5:
        raise SystemExit("Regression workflow R2 was below 0.5")
    return result


if __name__ == "__main__":
    main()
