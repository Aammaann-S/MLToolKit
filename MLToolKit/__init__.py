"""MLToolKit, an installable library of machine-learning implementations.

Importing the package does not import pandas or scikit-learn. The VS Code
extension reads the same catalog.json and inserts standalone code.
"""

from __future__ import annotations

import json
from importlib.resources import files

__version__ = "1.1.0"


def load_catalog() -> dict:
    """Return the snippet catalog shared with the VS Code extension."""
    text = files("MLToolKit").joinpath("catalog.json").read_text(encoding="utf-8")
    return json.loads(text)


__all__ = ["__version__", "load_catalog"]
