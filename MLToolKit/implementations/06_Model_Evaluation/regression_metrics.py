"""Public regression metrics used by evaluation workflows."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit_loader import load_impl

_metrics = load_impl("implementations/03_Regression/regression_evaluation.py")

regression_metrics = _metrics.regression_metrics
