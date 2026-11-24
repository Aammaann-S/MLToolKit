# Library compatibility

## Python

The code uses syntax supported by Python 3.10 and newer, including `list[str]` and `X | None`.

## Required libraries

Installed from `requirements.txt`:

| Library | Minimum | Used for |
| --- | --- | --- |
| numpy | 1.24 | Arrays and numeric transforms |
| pandas | 2.2 | Tables, CSV, Excel, JSON, SQL |
| scikit-learn | 1.3 | Models, pipelines, metrics, splits |
| matplotlib | 3.7 | Figures |
| seaborn | 0.13 | Statistical plots |
| scipy | 1.11 | Dendrograms, winsorization, normality tests |
| statsmodels | 0.14 | ARIMA |
| joblib | 1.3 | Model files |
| openpyxl | 3.1 | Excel reading and writing |
| pytest | 7.4 | Tests |

`OneHotEncoder` uses `sparse_output=False`. `DataFrame.fillna(method=...)` is not used; forward and backward fills call `ffill` and `bfill`. Datetime parts use `Series.dt.day_of_week`.

The local suite was executed with Python 3.14.6, pandas 3.1.1, and scikit-learn 1.9.1. GitHub Actions runs the same tests on Python 3.11 and 3.12.

## Optional libraries

See `requirements-optional.txt`. Importing a core module does not import PyTorch, TensorFlow, XGBoost, LightGBM, CatBoost, SHAP, LIME, imbalanced-learn, FastAPI, or MLflow. Calling one of those functions without the package raises `ImportError` and names the install command.

## Package import

The installable distribution name and the import name are both `MLToolKit`:

```python
import MLToolKit
```

## Editor

The TypeScript completion provider targets Visual Studio Code 1.85 and newer, including current Cursor builds. It activates for the `python` and `jupyter` language ids, and it only offers toolkit snippets after a real `MLToolKit` import in that file. Pip does not install the extension.
