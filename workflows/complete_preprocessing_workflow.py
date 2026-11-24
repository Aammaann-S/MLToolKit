"""Split first, then impute, encode, and scale using training statistics only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MLToolKit.toolkit_loader import load_impl


def main():
    missing = load_impl("implementations/01_Data_Preprocessing/missing_values.py")
    encoding = load_impl("implementations/01_Data_Preprocessing/categorical_encoding.py")
    scaling = load_impl("implementations/01_Data_Preprocessing/feature_scaling.py")
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "age": rng.normal(35, 8, 80),
            "income": rng.normal(50000, 5000, 80),
            "city": rng.choice(["Pune", "Delhi", "Chennai"], 80),
            "target": rng.integers(0, 2, 80),
        }
    )
    frame.loc[0:5, "age"] = np.nan
    train, test = train_test_split(frame, test_size=0.25, random_state=42, stratify=frame["target"])
    train_imputed, test_imputed, means = missing.impute_mean(train, test, columns=["age"])
    train_encoded, test_encoded, _encoder = encoding.one_hot_encode(train_imputed, test_imputed, columns=["city"])
    numeric = ["age", "income"]
    train_scaled, test_scaled, scaler = scaling.standard_scale(train_encoded, test_encoded, columns=numeric)
    print("train_rows", len(train_scaled))
    print("test_rows", len(test_scaled))
    print("scaler_mean_age", round(float(scaler.mean_[numeric.index("age")]), 4))
    if train_scaled["age"].isna().any() or test_scaled["age"].isna().any():
        raise SystemExit("Preprocessing workflow left missing ages")
    if abs(float(means["age"]) - float(train["age"].mean())) > 1e-8:
        raise SystemExit("Mean imputation used a value other than the training mean")
    return train_scaled, test_scaled


if __name__ == "__main__":
    main()
