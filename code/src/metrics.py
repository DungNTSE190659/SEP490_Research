"""Prediction metrics: AUC, RMSE, ACC. Thin wrappers over scikit-learn with
safe handling of degenerate folds (e.g. a fold with a single class)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # AUC is undefined if only one class is present in y_true.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_prob)))
    acc = float(((y_prob >= 0.5).astype(int) == y_true).mean())
    return {"auc": auc, "rmse": rmse, "acc": acc}


def aggregate_folds(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Given per-fold metric dicts, return {metric: {mean, std}}."""
    keys = fold_metrics[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out
