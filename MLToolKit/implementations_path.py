"""Import a sibling implementation file without turning numbered folders into packages."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_sibling(caller_file: str, filename: str):
    path = Path(caller_file).resolve().parent / filename
    module_name = "mla_sibling_" + path.parent.name + "_" + path.stem
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
