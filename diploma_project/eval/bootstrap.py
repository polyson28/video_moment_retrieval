from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


MetricFn = Callable[[pd.DataFrame], dict[str, float] | float]


def _query_column(df: pd.DataFrame) -> str:
    for candidate in ("qid", "source_qid", "query_id"):
        if candidate in df.columns:
            return candidate
    raise KeyError("bootstrap_ci requires one of qid/source_qid/query_id columns")


def _system_columns(df: pd.DataFrame) -> list[str]:
    candidates = [col for col in ["system_name", "system", "budget", "reject_type", "compression_policy", "window_lengths"] if col in df.columns]
    return candidates or ["__single_system__"]


def _as_metric_dict(value: dict[str, float] | float) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    return {"metric": float(value)}


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: MetricFn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Query-level bootstrap confidence intervals for each system.

    The function resamples query/example ids with replacement within each
    system group. `metric_fn` receives the resampled rows and should return
    either a scalar or a mapping of metric names to values.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if df.empty:
        return pd.DataFrame(columns=["metric", "mean", "ci_low", "ci_high"])

    frame = df.copy()
    query_col = _query_column(frame)
    system_cols = _system_columns(frame)
    if system_cols == ["__single_system__"]:
        frame["__single_system__"] = "all"

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    grouped = frame.groupby(system_cols, dropna=False, sort=False)
    quantiles = [alpha / 2.0, 1.0 - alpha / 2.0]

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = dict(zip(system_cols, keys, strict=True))
        query_ids = group[query_col].astype(str).drop_duplicates().to_numpy()
        by_query = {qid: part for qid, part in group.groupby(group[query_col].astype(str), sort=False)}

        original = _as_metric_dict(metric_fn(group))
        boot_values: dict[str, list[float]] = {name: [] for name in original}
        for _ in range(int(n_boot)):
            sampled = rng.choice(query_ids, size=len(query_ids), replace=True)
            sample_frame = pd.concat([by_query[str(qid)] for qid in sampled], ignore_index=True)
            metrics = _as_metric_dict(metric_fn(sample_frame))
            for name in original:
                boot_values[name].append(float(metrics.get(name, np.nan)))

        for name, values in boot_values.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            ci_low, ci_high = np.quantile(arr, quantiles) if len(arr) else (np.nan, np.nan)
            rows.append(
                {
                    **key_payload,
                    "metric": name,
                    "mean": float(original[name]),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_boot": int(n_boot),
                    "alpha": float(alpha),
                    "n_queries": int(len(query_ids)),
                }
            )

    result = pd.DataFrame(rows)
    if "__single_system__" in result.columns:
        result = result.drop(columns=["__single_system__"])
    return result

