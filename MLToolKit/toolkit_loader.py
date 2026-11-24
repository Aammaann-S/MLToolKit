"""Load implementation modules by file path.

Numbered folders are kept for browsing. They are not Python packages, so
workflows and tests import modules through this loader.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent


def load_impl(relative_path: str) -> ModuleType:
    """Import an implementation file such as ``implementations/03_Regression/ridge_regression.py``."""
    path = (ROOT / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = "mla_" + "_".join(path.relative_to(ROOT).with_suffix("").parts)
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
