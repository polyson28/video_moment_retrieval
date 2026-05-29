from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _metric_value(metrics: Mapping[str, float] | pd.Series, name: str, default: float = 0.0) -> float:
    if name in metrics:
        value = metrics[name]
    elif name == "BalancedOpenSet" and "BalancedOpenSet@0.5" in metrics:
        value = metrics["BalancedOpenSet@0.5"]
    else:
        value = default
    if pd.isna(value):
        return float(default)
    return float(value)


def balanced_open_set(metrics: Mapping[str, float] | pd.Series) -> float:
    """BalancedOpenSet = mean of end-to-end retrieval and ID/OOD rejection."""
    return float(
        np.mean(
            [
                _metric_value(metrics, "R1@0.5_e2e"),
                _metric_value(metrics, "RA_ID"),
                _metric_value(metrics, "RA_OOD"),
            ]
        )
    )


def balanced_open_set_id(metrics: Mapping[str, float] | pd.Series) -> float:
    """BalancedOpenSet_ID excludes OOD negatives from the open-set average."""
    return float(np.mean([_metric_value(metrics, "R1@0.5_e2e"), _metric_value(metrics, "RA_ID")]))


def weighted_open_set(
    metrics: Mapping[str, float] | pd.Series,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted open-set score over R1@0.5_e2e, RA_ID, and RA_OOD.

    The default is identical to BalancedOpenSet. Weight keys can be either
    metric names (`R1@0.5_e2e`, `RA_ID`, `RA_OOD`) or short names
    (`retrieval`, `id`, `ood`).
    """
    aliases = {
        "retrieval": "R1@0.5_e2e",
        "id": "RA_ID",
        "ood": "RA_OOD",
    }
    raw_weights = weights or {"R1@0.5_e2e": 1.0, "RA_ID": 1.0, "RA_OOD": 1.0}
    expanded = {aliases.get(str(key), str(key)): float(value) for key, value in raw_weights.items()}
    total = sum(expanded.values())
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    score = 0.0
    for metric_name, weight in expanded.items():
        score += weight * _metric_value(metrics, metric_name)
    return float(score / total)


# Public aliases matching the metric names used in experiment tables.
BalancedOpenSet = balanced_open_set
BalancedOpenSet_ID = balanced_open_set_id
WeightedOpenSet = weighted_open_set


def summarize_open_set_predictions(df: pd.DataFrame) -> dict[str, float]:
    """Compute canonical retrieval and rejection metrics from per-query predictions."""
    pos = df[df["sample_type"].eq("pos")]
    id_neg = df[df["sample_type"].eq("id_neg")]
    ood_neg = df[df["sample_type"].eq("ood_neg")]

    accepted = pos["accepted"].fillna(True).astype(bool).to_numpy() if "accepted" in pos else np.ones(len(pos), dtype=bool)
    id_rejected = id_neg["rejected"].fillna(False).astype(bool).to_numpy() if "rejected" in id_neg else np.zeros(len(id_neg), dtype=bool)
    ood_rejected = ood_neg["rejected"].fillna(False).astype(bool).to_numpy() if "rejected" in ood_neg else np.zeros(len(ood_neg), dtype=bool)
    iou = pos["iou"].to_numpy(dtype=float) if "iou" in pos else np.array([], dtype=float)

    metrics = {
        "R1@0.3": float((iou >= 0.3).mean()) if len(iou) else 0.0,
        "R1@0.5": float((iou >= 0.5).mean()) if len(iou) else 0.0,
        "R1@0.7": float((iou >= 0.7).mean()) if len(iou) else 0.0,
        "mean_iou": float(np.nanmean(iou)) if len(iou) else 0.0,
        "R1@0.5_e2e": float(((iou >= 0.5) & accepted).mean()) if len(iou) else 0.0,
        "RA_ID": float(id_rejected.mean()) if len(id_rejected) else 0.0,
        "RA_OOD": float(ood_rejected.mean()) if len(ood_rejected) else 0.0,
    }
    metrics["BalancedOpenSet"] = balanced_open_set(metrics)
    metrics["BalancedOpenSet_ID"] = balanced_open_set_id(metrics)
    return metrics
