from __future__ import annotations

import json
import os
import struct
import sys
import zlib
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from diploma_project.eval.bootstrap import bootstrap_ci
from diploma_project.eval.metrics import balanced_open_set_id, summarize_open_set_predictions


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "diploma_project").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root with AGENTS.md and diploma_project/")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diploma_project.experiments.joint_compression_reject as joint  # noqa: E402
from diploma_project.data_layer.schemas import Manifest  # noqa: E402


TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
SEEDS = [42, 43, 44]
MAIN_METRICS = ["R1@0.3", "R1@0.5", "R1@0.7", "mean_iou", "R1@0.5_e2e", "RA_ID", "RA_OOD", "BalancedOpenSet", "BalancedOpenSet_ID"]


def ensure_outputs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _write_simple_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3]) for y in range(height))
    payload = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(raw, level=9)),
        chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(payload))


def _draw_fallback_bar(path: Path, labels: Iterable[str], values: Iterable[float]) -> None:
    labels = list(labels)
    values = [float(v) for v in values]
    width, height = 900, 540
    pixels = bytearray([255, 255, 255] * width * height)

    def rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                idx = (y * width + x) * 3
                pixels[idx : idx + 3] = bytes(color)

    rect(72, 60, 76, height - 70, (30, 30, 30))
    rect(72, height - 74, width - 40, height - 70, (30, 30, 30))
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-12)
    slot = max(1, (width - 140) // max(1, len(values)))
    palette = [(58, 115, 178), (81, 163, 81), (222, 132, 56), (167, 86, 169)]
    for i, value in enumerate(values):
        bar_h = int((height - 150) * value / max_value)
        x0 = 92 + i * slot
        x1 = min(x0 + max(8, int(slot * 0.62)), width - 50)
        rect(x0, height - 74 - bar_h, x1, height - 74, palette[i % len(palette)])
    _write_simple_png(path, width, height, pixels)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_predictions(result_dir: Path, split: str = "val") -> pd.DataFrame:
    path = result_dir / "predictions" / f"{split}_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    parquet = result_dir / "predictions" / f"{split}_predictions.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    raise FileNotFoundError(f"no predictions for {split} in {result_dir}")


def metric_row(frame: pd.DataFrame, system_name: str, beta: float | None = None, **extra: Any) -> dict[str, Any]:
    metrics = summarize_open_set_predictions(frame)
    row = {
        "system": system_name,
        "beta": beta,
        **extra,
        **metrics,
    }
    row["BalancedOpenSet_ID"] = balanced_open_set_id(row)
    return row


def _select_system(df: pd.DataFrame, system: str, budget: float) -> pd.DataFrame:
    return df[df["system_name"].eq(system) & np.isclose(df["budget"].astype(float), budget)].copy()


def make_joint_main_qatopk() -> pd.DataFrame:
    exp6 = ROOT / "results" / "exp6_joint_compression_reject"
    exp6c = ROOT / "results" / "exp6c_joint_context_reserve_ablation"
    old_joint = read_csv(exp6 / "metrics" / "summary_by_system.csv")
    context = read_csv(exp6c / "metrics" / "summary_by_variant.csv")

    rows: list[pd.Series] = []

    def add_from(frame: pd.DataFrame, system_name: str, budget: float, label: str) -> None:
        match = frame[frame["system_name"].eq(system_name) & np.isclose(frame["budget"].astype(float), budget)]
        if match.empty:
            raise KeyError(f"missing {system_name} budget={budget}")
        row = match.iloc[0].copy()
        row["system"] = label
        row["beta"] = float(budget)
        rows.append(row)

    add_from(context, "full_no_reject", 1.0, "Full retrieval no reject")
    add_from(context, "full_learned_reject", 1.0, "Full retrieval + learned reject")
    for beta in [0.5, 0.25, 0.1]:
        add_from(old_joint, "uniform_compression_reject", beta, "Uniform compression + learned reject")
    for beta in [0.5, 0.25, 0.1]:
        add_from(context, "qa_topk_no_reject", beta, "QA-TopK compression no reject")
    for beta in [0.5, 0.25, 0.1]:
        add_from(context, "qa_topk_learned_reject", beta, "QA-TopK compression + learned reject")

    table = pd.DataFrame(rows)
    rename = {"BalancedOpenSet@0.5": "BalancedOpenSet"}
    table = table.rename(columns=rename)
    table["BalancedOpenSet_ID"] = table.apply(balanced_open_set_id, axis=1)
    keep = [
        "system",
        "system_name",
        "compression_policy",
        "beta",
        "reject_type",
        "R1@0.5_before_reject",
        "R1@0.7_before_reject",
        "mean_iou",
        "R1@0.5_e2e",
        "R1@0.7_e2e",
        "RA_ID",
        "RA_OOD",
        "BalancedOpenSet",
        "BalancedOpenSet_ID",
        "compression_ratio",
        "avg_inference_time_per_query",
    ]
    table = table[[col for col in keep if col in table.columns]]
    table.to_csv(TABLE_DIR / "joint_main_qatopk.csv", index=False)
    return table


def _split_train_calibration(train_df: pd.DataFrame, seed: int) -> tuple[pd.Index, pd.Index]:
    train_idx, cal_idx = train_test_split(
        train_df.index,
        test_size=float(joint.CONFIG["calibration_size"]),
        random_state=int(seed),
        stratify=train_df["sample_type"].astype(str),
    )
    return pd.Index(train_idx), pd.Index(cal_idx)


def _choose_threshold(calibration: pd.DataFrame, score_col: str = "confidence_prob") -> dict[str, float]:
    scores = calibration[score_col].to_numpy(dtype=float)
    candidates = np.unique(scores)
    candidates = np.concatenate([[-np.inf], candidates, [np.nextafter(candidates.max(), np.inf)]])
    best = None
    pos = calibration["sample_type"].eq("pos").to_numpy()
    id_neg = calibration["sample_type"].eq("id_neg").to_numpy()
    ood_neg = calibration["sample_type"].eq("ood_neg").to_numpy()
    for threshold in candidates:
        rejected = scores < float(threshold)
        pos_accept = float((~rejected[pos]).mean()) if pos.any() else 0.0
        ra_id = float(rejected[id_neg].mean()) if id_neg.any() else 0.0
        ra_ood = float(rejected[ood_neg].mean()) if ood_neg.any() else 0.0
        item = (float(np.mean([pos_accept, ra_id, ra_ood])), pos_accept, ra_id, ra_ood, float(threshold))
        if best is None or item > best:
            best = item
    assert best is not None
    return {"threshold": best[4], "calibration_balanced": best[0], "calibration_pos_accept": best[1], "calibration_RA_ID": best[2], "calibration_RA_OOD": best[3]}


def apply_seeded_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    fit_idx, cal_idx = _split_train_calibration(train_df, seed)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(train_df.loc[fit_idx, joint.REJECT_FEATURES])
    y_fit = train_df.loc[fit_idx, "y_present"].to_numpy()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(seed))
    clf.fit(x_fit, y_fit)
    for frame in (train_df, val_df):
        frame["confidence_prob"] = clf.predict_proba(scaler.transform(frame[joint.REJECT_FEATURES]))[:, 1]
        frame["classifier_score"] = frame["confidence_prob"]
    threshold_info = _choose_threshold(train_df.loc[cal_idx], "confidence_prob")
    threshold = float(threshold_info["threshold"])
    val_df["reject_threshold"] = threshold
    val_df["rejected"] = val_df["confidence_prob"].to_numpy(dtype=float) < threshold
    val_df["accepted"] = ~val_df["rejected"]
    return val_df, threshold_info


def _main_prediction_sources() -> list[dict[str, Any]]:
    return [
        {"result_dir": ROOT / "results" / "exp6c_joint_context_reserve_ablation", "system_name": "full_no_reject", "budget": 1.0, "label": "Full retrieval no reject", "reject": False},
        {"result_dir": ROOT / "results" / "exp6c_joint_context_reserve_ablation", "system_name": "full_learned_reject", "budget": 1.0, "label": "Full retrieval + learned reject", "reject": True},
        *[
            {"result_dir": ROOT / "results" / "exp6_joint_compression_reject", "system_name": "uniform_compression_reject", "budget": beta, "label": "Uniform compression + learned reject", "reject": True}
            for beta in [0.5, 0.25, 0.1]
        ],
        *[
            {"result_dir": ROOT / "results" / "exp6c_joint_context_reserve_ablation", "system_name": "qa_topk_no_reject", "budget": beta, "label": "QA-TopK compression no reject", "reject": False}
            for beta in [0.5, 0.25, 0.1]
        ],
        *[
            {"result_dir": ROOT / "results" / "exp6c_joint_context_reserve_ablation", "system_name": "qa_topk_learned_reject", "budget": beta, "label": "QA-TopK compression + learned reject", "reject": True}
            for beta in [0.5, 0.25, 0.1]
        ],
    ]


def make_joint_main_by_seed() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    prediction_cache: dict[Path, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for source in _main_prediction_sources():
        result_dir = Path(source["result_dir"])
        if result_dir not in prediction_cache:
            prediction_cache[result_dir] = (read_predictions(result_dir, "train"), read_predictions(result_dir, "val"))
        train_all, val_all = prediction_cache[result_dir]
        train_src = _select_system(train_all, str(source["system_name"]), float(source["budget"]))
        val_src = _select_system(val_all, str(source["system_name"]), float(source["budget"]))
        if train_src.empty or val_src.empty:
            raise KeyError(f"missing predictions for {source}")
        for seed in SEEDS:
            if source["reject"]:
                val_eval, threshold = apply_seeded_classifier(train_src, val_src, seed)
            else:
                val_eval = val_src.copy()
                val_eval["accepted"] = True
                val_eval["rejected"] = False
                val_eval["confidence_prob"] = np.nan
                threshold = {"threshold": np.nan}
            row = metric_row(val_eval, str(source["label"]), float(source["budget"]), seed=seed, threshold=threshold["threshold"])
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(TABLE_DIR / "joint_main_qatopk_by_seed.csv", index=False)

    summary_rows = []
    for (system, beta), group in per_seed.groupby(["system", "beta"], dropna=False, sort=False):
        for metric in MAIN_METRICS:
            values = group[metric].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "system": system,
                    "beta": beta,
                    "metric": metric,
                    "metric_mean": float(np.mean(values)),
                    "metric_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "metric_ci_low": float(np.quantile(values, 0.025)),
                    "metric_ci_high": float(np.quantile(values, 0.975)),
                    "n_seeds": int(len(values)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "joint_main_qatopk_seed_summary.csv", index=False)
    return per_seed, summary


def make_bootstrap_ci_tables(n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for source in _main_prediction_sources():
        val = read_predictions(Path(source["result_dir"]), "val")
        data = _select_system(val, str(source["system_name"]), float(source["budget"]))
        if not source["reject"]:
            data = data.copy()
            data["accepted"] = True
            data["rejected"] = False

        original = summarize_open_set_predictions(data)
        pos = data[data["sample_type"].eq("pos")]
        id_neg = data[data["sample_type"].eq("id_neg")]
        ood_neg = data[data["sample_type"].eq("ood_neg")]
        pos_iou = pos["iou"].to_numpy(dtype=float)
        pos_acc = pos["accepted"].astype(bool).to_numpy() if "accepted" in pos else np.ones(len(pos), dtype=bool)
        id_rej = id_neg["rejected"].astype(bool).to_numpy() if "rejected" in id_neg else np.zeros(len(id_neg), dtype=bool)
        ood_rej = ood_neg["rejected"].astype(bool).to_numpy() if "rejected" in ood_neg else np.zeros(len(ood_neg), dtype=bool)

        boot = {metric: [] for metric in MAIN_METRICS}
        for _ in range(int(n_boot)):
            pidx = rng.integers(0, len(pos_iou), size=len(pos_iou)) if len(pos_iou) else np.array([], dtype=int)
            iidx = rng.integers(0, len(id_rej), size=len(id_rej)) if len(id_rej) else np.array([], dtype=int)
            oidx = rng.integers(0, len(ood_rej), size=len(ood_rej)) if len(ood_rej) else np.array([], dtype=int)
            r03 = float((pos_iou[pidx] >= 0.3).mean()) if len(pidx) else 0.0
            r05 = float((pos_iou[pidx] >= 0.5).mean()) if len(pidx) else 0.0
            r07 = float((pos_iou[pidx] >= 0.7).mean()) if len(pidx) else 0.0
            miou = float(np.nanmean(pos_iou[pidx])) if len(pidx) else 0.0
            e2e = float(((pos_iou[pidx] >= 0.5) & pos_acc[pidx]).mean()) if len(pidx) else 0.0
            ra_id = float(id_rej[iidx].mean()) if len(iidx) else 0.0
            ra_ood = float(ood_rej[oidx].mean()) if len(oidx) else 0.0
            values = {
                "R1@0.3": r03,
                "R1@0.5": r05,
                "R1@0.7": r07,
                "mean_iou": miou,
                "R1@0.5_e2e": e2e,
                "RA_ID": ra_id,
                "RA_OOD": ra_ood,
                "BalancedOpenSet": float(np.mean([e2e, ra_id, ra_ood])),
                "BalancedOpenSet_ID": float(np.mean([e2e, ra_id])),
            }
            for metric, value in values.items():
                boot[metric].append(value)

        for metric in MAIN_METRICS:
            arr = np.asarray(boot[metric], dtype=float)
            rows.append(
                {
                    "system": str(source["label"]),
                    "beta": float(source["budget"]),
                    "metric": metric,
                    "mean": float(original[metric]),
                    "ci_low": float(np.quantile(arr, 0.025)),
                    "ci_high": float(np.quantile(arr, 0.975)),
                    "n_boot": int(n_boot),
                }
            )
    ci = pd.DataFrame(rows)
    ci.to_csv(TABLE_DIR / "joint_main_qatopk_bootstrap_ci.csv", index=False)
    return ci


def load_manifest() -> Manifest:
    manifest_path = ROOT / "results" / "exp6_joint_compression_reject" / "manifest.json"
    return Manifest.load(manifest_path)


def make_ood_triviality_analysis() -> pd.DataFrame:
    ensure_outputs()
    manifest = load_manifest()
    sample_rows = []
    for split in ["val_pos", "val_id_neg", "val_ood_neg"]:
        for sample in manifest.splits[split]:
            sample_rows.append(
                {
                    "split": split,
                    "sample_type": sample.label_type,
                    "qid": str(sample.qid),
                    "source_qid": str(sample.meta.get("source_qid", sample.qid)),
                    "query_length": len(sample.query.split()),
                    "query": sample.query,
                }
            )
    samples = pd.DataFrame(sample_rows)
    pos_sources = set(samples[samples["sample_type"].eq("pos")]["source_qid"])
    ood_sources = set(samples[samples["sample_type"].eq("ood_neg")]["source_qid"])

    ra_rows = []
    for label, path in [
        ("notebook_06_joint", ROOT / "results" / "exp6_joint_compression_reject" / "metrics" / "summary_by_system.csv"),
        ("notebook_07_reject_type", ROOT / "results" / "exp6b_joint_reject_type_ablation" / "metrics" / "summary_by_system.csv"),
        ("notebook_08_context_reserve", ROOT / "results" / "exp6c_joint_context_reserve_ablation" / "metrics" / "summary_by_system.csv"),
    ]:
        if path.exists():
            summary = pd.read_csv(path)
            summary = summary[summary["reject_type"].astype(str).ne("none")]
            for _, row in summary.iterrows():
                ra_rows.append({"notebook": label, "system": row["system_name"], "beta": row["budget"], "RA_OOD": row["RA_OOD"]})
    ra = pd.DataFrame(ra_rows)
    query_stats = samples.groupby("sample_type")["query_length"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    rows = [
        {"analysis": "source_qid_overlap_pos_ood", "value": len(pos_sources & ood_sources), "detail": "positive/OOD source_qid overlap count"},
        {"analysis": "n_positive_source_qid", "value": len(pos_sources), "detail": "unique positive source_qid"},
        {"analysis": "n_ood_source_qid", "value": len(ood_sources), "detail": "unique OOD source_qid"},
        {"analysis": "all_ra_ood_is_one", "value": bool((ra["RA_OOD"] == 1.0).all()) if not ra.empty else np.nan, "detail": "all saved configurations have RA_OOD=1.0"},
    ]
    for _, row in query_stats.iterrows():
        for col in ["count", "mean", "std", "min", "median", "max"]:
            rows.append({"analysis": f"query_length_{col}_{row['sample_type']}", "value": row[col], "detail": "query length by sample_type"})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "ood_triviality.csv", index=False)

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        samples.boxplot(column="query_length", by="sample_type", ax=ax)
        ax.set_title("Query length by type")
        ax.set_xlabel("Sample type")
        ax.set_ylabel("Query length, words")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "query_length_by_type.png", dpi=300)
        plt.close(fig)
    except ModuleNotFoundError:
        means = samples.groupby("sample_type")["query_length"].mean().sort_index()
        _draw_fallback_bar(FIGURE_DIR / "query_length_by_type.png", means.index, means.values)
    return table


def make_confidence_scale_diagnostics() -> pd.DataFrame:
    ensure_outputs()
    rows = []
    configs = [
        ("06_joint", ROOT / "results" / "exp6_joint_compression_reject"),
        ("07_reject_type", ROOT / "results" / "exp6b_joint_reject_type_ablation"),
        ("08_context_reserve", ROOT / "results" / "exp6c_joint_context_reserve_ablation"),
    ]
    for notebook, result_dir in configs:
        threshold_path = result_dir / "reject" / "thresholds.json"
        if not threshold_path.exists():
            continue
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        val = read_predictions(result_dir, "val")
        for (system_name, budget, reject_type), group in val.groupby(["system_name", "budget", "reject_type"], dropna=False):
            if str(reject_type) == "none":
                continue
            if str(reject_type) == "top_score_threshold":
                score_col = "top_window_score"
                source = "top_window_score"
            else:
                score_col = "confidence_prob" if "confidence_prob" in group.columns else "classifier_score"
                source = "predict_proba[:, 1]" if str(reject_type) == "confidence_classifier" else score_col
            scores = group[score_col].dropna().astype(float)
            th = group["reject_threshold"].dropna().astype(float) if "reject_threshold" in group else pd.Series(dtype=float)
            rows.append(
                {
                    "notebook": notebook,
                    "system": system_name,
                    "reject_type": reject_type,
                    "confidence_source": source,
                    "threshold_min": float(th.min()) if len(th) else np.nan,
                    "threshold_max": float(th.max()) if len(th) else np.nan,
                    "score_min": float(scores.min()) if len(scores) else np.nan,
                    "score_max": float(scores.max()) if len(scores) else np.nan,
                    "threshold_records": len(thresholds),
                }
            )
    table = pd.DataFrame(rows).sort_values(["notebook", "reject_type", "system"])
    table.to_csv(TABLE_DIR / "confidence_scale_diagnostics.csv", index=False)
    return table


def _selected_indices_random(sample: Any, video_features: np.ndarray, query_feature: np.ndarray, policy: str, budget: float, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if policy != "random":
        return joint.select_indices(sample, video_features, query_feature, policy, budget)
    t = int(video_features.shape[0])
    k = joint.budget_to_k(t, budget)
    rng_seed = abs(hash((str(sample.qid), str(sample.vid), int(seed)))) % (2**32)
    rng = np.random.default_rng(rng_seed)
    indices = np.sort(rng.choice(np.arange(t), size=k, replace=False).astype(np.int64))
    return indices, {
        "method": "random",
        "budget": float(budget),
        "num_original_tokens": t,
        "num_selected_tokens": int(len(indices)),
        "retained_token_ratio": float(len(indices) / t),
        "compression_ratio": float(t / len(indices)),
        "selected_indices": indices.tolist(),
    }


def _make_min_window_selector(min_tokens: int, base_selector):
    def selector(sample: Any, video_features: np.ndarray, query_feature: np.ndarray, policy: str, budget: float) -> tuple[np.ndarray, dict[str, Any]]:
        t = int(video_features.shape[0])
        effective_budget = max(float(budget), min(1.0, float(min_tokens) / max(t, 1)))
        indices, metadata = base_selector(sample, video_features, query_feature, policy, effective_budget)
        metadata = dict(metadata)
        metadata["requested_budget"] = float(budget)
        metadata["effective_budget_for_fixed_window"] = float(effective_budget)
        return indices, metadata

    return selector


def make_fixed_window_control(run_inference: bool = True) -> pd.DataFrame:
    ensure_outputs()
    baseline = read_predictions(ROOT / "results" / "exp6c_joint_context_reserve_ablation", "val")
    baseline = _select_system(baseline, "full_no_reject", 1.0)
    pos = baseline[baseline["sample_type"].eq("pos")].copy()
    dist = pos["pred_length_tokens"].value_counts(normalize=True).rename("fraction").reset_index().rename(columns={"index": "length_tokens", "pred_length_tokens": "length_tokens"})
    dist["mean_predicted_duration"] = float(pos["predicted_duration"].mean())
    dist["mean_gt_duration"] = float((pos["gt_end"] - pos["gt_start"]).mean())
    dist.to_csv(TABLE_DIR / "window_length_distribution.csv", index=False)

    counts = pos["pred_length_tokens"].value_counts().sort_index()
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Predicted window length, tokens")
        ax.set_ylabel("Count")
        ax.set_title("Predicted window length distribution")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "predicted_window_length_distribution.png", dpi=300)
        plt.close(fig)
    except ModuleNotFoundError:
        _draw_fallback_bar(FIGURE_DIR / "predicted_window_length_distribution.png", counts.index.astype(str), counts.values)

    if not run_inference:
        return pd.DataFrame()

    manifest = load_manifest()
    train_samples = manifest.splits["train_mixed"]
    val_samples = [*manifest.splits["val_pos"], *manifest.splits["val_id_neg"], *manifest.splits["val_ood_neg"]]
    head = joblib.load(ROOT / joint.CONFIG["retrieval_head_source"])
    scaler = joblib.load(ROOT / joint.CONFIG["retrieval_scaler_source"])
    original_window_lengths = list(joint.CONFIG["window_lengths_tokens"])
    original_method = joint.CONFIG["query_aware_method"]
    original_context = joint.CONFIG["context_radius"]
    original_reserve = joint.CONFIG["reserve_fraction"]
    original_selector = joint.select_indices
    rows = []
    try:
        joint.CONFIG["query_aware_method"] = "qa_topk"
        joint.CONFIG["context_radius"] = 0
        joint.CONFIG["reserve_fraction"] = 0.0
        for lengths in [[8], [16], [32], [4, 8, 16], [4, 8, 16, 32]]:
            joint.CONFIG["window_lengths_tokens"] = lengths
            joint.select_indices = _make_min_window_selector(max(lengths), original_selector)
            specs = [
                joint.SystemSpec("Full retrieval", "none", 1.0, "confidence_classifier"),
                joint.SystemSpec("Uniform beta=0.5", "uniform", 0.5, "confidence_classifier"),
                joint.SystemSpec("QA-TopK beta=0.5", "query_aware", 0.5, "confidence_classifier"),
                joint.SystemSpec("QA-TopK beta=0.25", "query_aware", 0.25, "confidence_classifier"),
            ]
            for spec in specs:
                train_df = joint.score_samples(train_samples, head, scaler, spec, f"train_fixed_{lengths}")
                val_df = joint.score_samples(val_samples, head, scaler, spec, f"val_fixed_{lengths}")
                _, val_eval, threshold = joint.apply_reject(train_df, val_df, spec)
                row = metric_row(val_eval, spec.system_name, spec.budget, window_lengths=",".join(map(str, lengths)), threshold=threshold.get("threshold", np.nan))
                rows.append(row)
        result = pd.DataFrame(rows)
        result.to_csv(TABLE_DIR / "fixed_window_control.csv", index=False)
        return result
    finally:
        joint.CONFIG["window_lengths_tokens"] = original_window_lengths
        joint.CONFIG["query_aware_method"] = original_method
        joint.CONFIG["context_radius"] = original_context
        joint.CONFIG["reserve_fraction"] = original_reserve
        joint.select_indices = original_selector


def make_nonadaptive_extra_baselines() -> pd.DataFrame:
    ensure_outputs()
    rows = []
    metrics_path = ROOT / "results" / "exp2_compression_positive_only_qa" / "metrics_per_seed.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for name in ["random", "video_only_saliency"]:
            subset = metrics[metrics.astype(str).apply(lambda col: col.str.contains(name, case=False, regex=False)).any(axis=1)]
            if subset.empty:
                rows.append({"baseline": name, "status": "not_found_in_metrics_per_seed", "note": "Not included in final protocol."})
            else:
                numeric = subset.select_dtypes(include=[np.number])
                item = {"baseline": name, "status": "available", "rows": int(len(subset))}
                for col in numeric.columns:
                    item[f"{col}_mean"] = float(numeric[col].mean())
                    item[f"{col}_std"] = float(numeric[col].std(ddof=1)) if len(numeric[col].dropna()) > 1 else 0.0
                rows.append(item)
    else:
        rows.append({"baseline": "random", "status": "not_found", "note": "Random keep is not part of the final joint protocol."})
        rows.append({"baseline": "video_only_saliency", "status": "not_found", "note": "Video-only saliency is not part of the final joint protocol."})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "nonadaptive_extra_baselines.csv", index=False)
    return table


def run_all(lightweight: bool = False) -> dict[str, pd.DataFrame]:
    ensure_outputs()
    outputs = {
        "joint_main_qatopk": make_joint_main_qatopk(),
        "joint_main_by_seed": make_joint_main_by_seed()[0],
        "bootstrap_ci": make_bootstrap_ci_tables(n_boot=200 if lightweight else 1000),
        "ood_triviality": make_ood_triviality_analysis(),
        "confidence_scale_diagnostics": make_confidence_scale_diagnostics(),
        "nonadaptive_extra_baselines": make_nonadaptive_extra_baselines(),
    }
    outputs["fixed_window_control"] = make_fixed_window_control(run_inference=not lightweight)
    return outputs


if __name__ == "__main__":
    run_all(lightweight="--lightweight" in sys.argv)
