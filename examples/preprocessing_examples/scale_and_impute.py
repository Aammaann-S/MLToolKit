"""Exam-style preprocessing. Statistics come from the training split only."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
df = pd.DataFrame({"feature": rng.normal(size=40), "target": rng.normal(size=40)})
df.loc[0:3, "feature"] = np.nan

X = df[["feature"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train_prepared = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_prepared = scaler.transform(imputer.transform(X_test))
print(X_train_prepared.shape, X_test_prepared.shape)
