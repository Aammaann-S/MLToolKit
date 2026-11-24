"""Optional SHAP and LIME explanations.

Both libraries are optional. The functions raise ImportError with an install
hint when the package is missing, and they do not run at import time.
"""

from __future__ import annotations


def shap_summary(model, X):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install SHAP with: pip install shap") from exc
    explainer = shap.Explainer(model, X)
    return explainer(X)


def lime_explain_row(model, X_train, row, feature_names: list[str], class_names: list[str] | None = None):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError("Install LIME with: pip install lime") from exc
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )
    return explainer.explain_instance(row, model.predict_proba)
