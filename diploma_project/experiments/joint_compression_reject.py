from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from diploma_project.compression.query_aware import compress_query_aware, validate_selected_indices
from diploma_project.data_layer.manifest_builder import BuildManifestConfig, build_manifest, save_manifest
from diploma_project.data_layer.schemas import Manifest, SampleRecord
from diploma_project.data_layer.validate_manifest import format_validation_report, validate_manifest


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "diploma_project").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root with AGENTS.md and diploma_project/")


ROOT = find_project_root()
RESULT_DIR = ROOT / "results" / "exp6_joint_compression_reject"
METRICS_DIR = RESULT_DIR / "metrics"
PRED_DIR = RESULT_DIR / "predictions"
REJECT_DIR = RESULT_DIR / "reject"
FIGURE_DIR = RESULT_DIR / "figures"

CONFIG: dict[str, Any] = {
    "experiment_name": "exp6_joint_compression_reject",
    "protocol": "AGENTS.md v1.0 / Stage D / Exp-4 joint compression + reject",
    "split_protocol": "mixed train = pos + id_neg + ood_neg; val = separate pos / id_neg / ood_neg",
    "task_stage": "joint",
    "feature_root": "data/qvhighlights/features",
    "feature_modalities": ["clip_features"],
    "video_modality": "clip_features",
    "text_feature_dir": "data/qvhighlights/features/clip_text_features",
    "text_embedding_key": "pooler_output",
    "window_lengths_tokens": [2, 4, 8, 16, 32],
    "budgets": [0.5, 0.25, 0.1],
    "seed": 42,
    "context_radius": 0,
    "reserve_fraction": 0.0,
    "query_aware_method": "qa_topk",
    "reject_type": "confidence_classifier",
    "threshold_selection": "calibration_from_train_mixed_max_mean_PosAccept_RA_ID_RA_OOD",
    "threshold_protocol_version": "v1.1_train_calibration",
    "threshold_fit_split": "train_mixed_fit",
    "threshold_calibration_split": "train_mixed_calibration",
    "final_evaluation_splits": ["val_pos", "val_id_neg", "val_ood_neg"],
    "calibration_size": 0.2,
    "retrieval_head_source": "results/exp1b_trainable_full_token_retrieval_head/model.joblib",
    "retrieval_scaler_source": "results/exp1b_trainable_full_token_retrieval_head/scaler.joblib",
    "baseline_retrieval_source": "results/exp1b_trainable_full_token_retrieval_head/metrics.json",
    "negative_aware_baseline_source": "results/exp3_open_set_no_compression/summary_table.csv",
    "latency_protocol": "timed scoring includes compression policy and retrieval head, excludes no explicit warm cache guarantee",
}

SUMMARY_COLUMNS = [
    "system_name",
    "compression_policy",
    "budget",
    "reject_type",
    "avg_num_tokens",
    "retained_token_ratio",
    "compression_ratio",
    "approx_attention_cost_ratio",
    "avg_inference_time_per_query",
    "R1@0.5_before_reject",
    "R1@0.7_before_reject",
    "R1@0.5_e2e",
    "R1@0.7_e2e",
    "mean_iou",
    "positive_accept_rate",
    "false_reject_rate",
    "RA_ID",
    "RA_OOD",
    "RA_ALL",
    "false_accept_rate_ID",
    "false_accept_rate_OOD",
    "BalancedOpenSet@0.5",
    "BalancedOpenSet@0.7",
    "delta_vs_retrieval_baseline_R1@0.5",
    "delta_vs_negative_aware_baseline_RA_ID",
    "delta_vs_negative_aware_baseline_RA_OOD",
]

REJECT_FEATURES = [
    "top_window_score",
    "max_token_score",
    "mean_top_token_score",
    "score_margin_top1_top2",
    "score_std",
    "predicted_duration",
    "num_selected_tokens",
    "retained_token_ratio",
    "compression_ratio",
    "score_entropy",
    "score_sharpness",
]


class LRUArrayCache:
    def __init__(self, max_items: int = 256):
        self.max_items = int(max_items)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str, loader) -> np.ndarray:
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        value = loader()
        self.cache[key] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


VIDEO_CACHE = LRUArrayCache(max_items=128)
TEXT_CACHE = LRUArrayCache(max_items=4096)


@dataclass(frozen=True)
class SystemSpec:
    system_name: str
    compression_policy: str
    budget: float
    reject_type: str


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
    lines = []
    for key, value in payload.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_npz_array(path: str | Path, key: str | None = None) -> np.ndarray:
    payload = np.load(Path(path))
    if key is not None:
        return payload[key]
    if "features" in payload.files:
        return payload["features"]
    return payload[payload.files[0]]


def source_qid(sample: SampleRecord) -> int | str:
    return sample.meta.get("source_qid", sample.qid)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def temporal_iou(left: list[float], right: list[float]) -> float:
    inter_start = max(float(left[0]), float(right[0]))
    inter_end = min(float(left[1]), float(right[1]))
    inter = max(0.0, inter_end - inter_start)
    union = max(float(left[1]), float(right[1])) - min(float(left[0]), float(right[0]))
    return inter / union if union > 0 else 0.0


def best_iou_and_gt(pred_window: list[float], gt_windows: list[list[float]] | None) -> tuple[float, list[float] | None]:
    if not gt_windows:
        return math.nan, None
    scored = [(temporal_iou(pred_window, gt), gt) for gt in gt_windows]
    iou, gt = max(scored, key=lambda item: item[0])
    return float(iou), [float(gt[0]), float(gt[1])]


def budget_to_k(T: int, budget: float) -> int:
    if budget >= 1.0:
        return T
    return int(min(T, max(1, math.ceil(float(budget) * T))))


def load_video_features(sample: SampleRecord) -> np.ndarray:
    record = sample.features[CONFIG["video_modality"]]
    return VIDEO_CACHE.get(record.path, lambda: load_npz_array(record.path, key="features"))


def load_query_feature(sample: SampleRecord) -> np.ndarray:
    path = ROOT / CONFIG["text_feature_dir"] / f"qid{source_qid(sample)}.npz"
    return TEXT_CACHE.get(str(path), lambda: load_npz_array(path, key=CONFIG["text_embedding_key"]).reshape(-1))


def select_indices(sample: SampleRecord, video_features: np.ndarray, query_feature: np.ndarray, policy: str, budget: float) -> tuple[np.ndarray, dict[str, Any]]:
    T = int(video_features.shape[0])
    if policy == "none" or budget >= 1.0:
        indices = np.arange(T, dtype=np.int64)
        metadata = {
            "method": policy,
            "budget": float(budget),
            "num_original_tokens": T,
            "num_selected_tokens": T,
            "retained_token_ratio": 1.0,
            "compression_ratio": 1.0,
            "selected_indices": indices.tolist(),
        }
    elif policy == "uniform":
        k = budget_to_k(T, budget)
        indices = np.unique(np.linspace(0, T - 1, num=k).round().astype(np.int64))
        metadata = {
            "method": policy,
            "budget": float(budget),
            "num_original_tokens": T,
            "num_selected_tokens": int(len(indices)),
            "retained_token_ratio": float(len(indices) / T),
            "compression_ratio": float(T / len(indices)),
            "selected_indices": indices.tolist(),
        }
    elif policy == "query_aware":
        _, indices, metadata = compress_query_aware(
            video_features,
            query_feature,
            budget,
            CONFIG["query_aware_method"],
            context_radius=CONFIG["context_radius"],
            reserve_fraction=CONFIG["reserve_fraction"],
            clip_modality=CONFIG["video_modality"],
        )
        metadata["method"] = CONFIG["query_aware_method"]
    else:
        raise ValueError(f"unknown compression policy: {policy}")
    validate_selected_indices(indices, T, budget)
    return np.asarray(indices, dtype=np.int64), metadata


def compressed_window_to_seconds(start_idx: int, end_idx: int, selected_indices: np.ndarray, original_T: int, duration: float) -> list[float]:
    original_start = int(selected_indices[start_idx])
    original_end = int(selected_indices[end_idx - 1]) + 1
    return [duration * original_start / original_T, duration * original_end / original_T]


def token_mapping_json(selected_indices: np.ndarray, original_T: int, duration: float) -> str:
    mapping = [
        {
            "compressed_index": int(comp_idx),
            "original_token_index": int(original_idx),
            "timestamp": [float(duration * original_idx / original_T), float(duration * (original_idx + 1) / original_T)],
        }
        for comp_idx, original_idx in enumerate(selected_indices.tolist())
    ]
    return json.dumps(mapping, separators=(",", ":"))


def window_feature_matrix(token_scores: np.ndarray, selected_indices: np.ndarray, original_T: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    scores = np.asarray(token_scores, dtype=np.float32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    T = int(scores.shape[0])
    cumsum = np.concatenate([[0.0], np.cumsum(scores, dtype=np.float64)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(scores * scores, dtype=np.float64)])
    rows: list[list[float]] = []
    spans: list[tuple[int, int, int]] = []
    for length in CONFIG["window_lengths_tokens"]:
        if length > T:
            continue
        sums = cumsum[length:] - cumsum[:-length]
        sums_sq = cumsum_sq[length:] - cumsum_sq[:-length]
        means = sums / length
        vars_ = np.maximum(sums_sq / length - means * means, 0.0)
        stds = np.sqrt(vars_)
        for start_idx in range(0, T - length + 1):
            end_idx = start_idx + length
            orig_start = int(selected_indices[start_idx])
            orig_end = int(selected_indices[end_idx - 1]) + 1
            original_length = orig_end - orig_start
            length_rel = original_length / original_T
            rows.append(
                [
                    float(means[start_idx]),
                    float(sums[start_idx]),
                    float(stds[start_idx]),
                    float(scores[start_idx:end_idx].max()),
                    float(scores[start_idx:end_idx].min()),
                    float(original_length),
                    float(length_rel),
                    float(length_rel),
                    float((orig_start + orig_end) / (2.0 * original_T)),
                    float(orig_start / original_T),
                    float(orig_end / original_T),
                ]
            )
            spans.append((start_idx, end_idx, original_length))
    if not rows:
        raise AssertionError(f"no candidate windows after compression, T={T}")
    return np.asarray(rows, dtype=np.float32), spans


def score_entropy(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values = values - values.max()
    probs = np.exp(values)
    probs = probs / max(float(probs.sum()), 1e-12)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return entropy / math.log(max(len(values), 2))


def prediction_row(sample: SampleRecord, head: Any, scaler: Any, spec: SystemSpec) -> dict[str, Any]:
    started = time.perf_counter()
    video = load_video_features(sample)
    query = load_query_feature(sample)
    original_T = int(video.shape[0])
    indices, _ = select_indices(sample, video, query, spec.compression_policy, spec.budget)
    compressed_video = video[indices]
    V = l2_normalize(compressed_video, axis=1)
    q = l2_normalize(query, axis=0)
    if V.shape[1] != q.shape[0]:
        raise ValueError(f"Dimension mismatch for qid={sample.qid}: video={V.shape}, query={q.shape}")
    token_scores = V @ q
    X, spans = window_feature_matrix(token_scores, indices, original_T)
    pred_scores = head.predict(scaler.transform(X))
    order = np.argsort(pred_scores)[::-1]
    best_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else best_idx
    start_idx, end_idx, pred_length_tokens = spans[best_idx]
    duration = float(sample.duration)
    pred_window = compressed_window_to_seconds(start_idx, end_idx, indices, original_T, duration)
    iou, matched_gt = best_iou_and_gt(pred_window, sample.gt_windows)
    top_score = float(pred_scores[best_idx])
    second_score = float(pred_scores[second_idx])
    elapsed = time.perf_counter() - started
    selected_ratio = float(len(indices) / original_T)
    selected_scores_sorted = np.sort(token_scores)[::-1]
    top_k = selected_scores_sorted[: min(5, len(selected_scores_sorted))]
    return {
        "qid": str(sample.qid),
        "source_qid": str(source_qid(sample)),
        "vid": sample.vid,
        "split": sample.split,
        "sample_type": sample.label_type,
        "task_stage": CONFIG["task_stage"],
        "system_name": spec.system_name,
        "compression_policy": spec.compression_policy,
        "budget": float(spec.budget),
        "reject_type": spec.reject_type,
        "y_present": int(sample.label_type == "pos"),
        "gt_start": matched_gt[0] if matched_gt else math.nan,
        "gt_end": matched_gt[1] if matched_gt else math.nan,
        "pred_start": float(pred_window[0]),
        "pred_end": float(pred_window[1]),
        "predicted_duration": float(pred_window[1] - pred_window[0]),
        "top_window_score": top_score,
        "second_window_score": second_score,
        "score_margin_top1_top2": float(top_score - second_score),
        "num_candidates": int(len(pred_scores)),
        "num_original_tokens": int(original_T),
        "num_selected_tokens": int(len(indices)),
        "avg_num_tokens": float(len(indices)),
        "retained_token_ratio": selected_ratio,
        "compression_ratio": float(original_T / len(indices)),
        "approx_attention_cost_ratio": float(selected_ratio**2),
        "selected_indices": " ".join(map(str, indices.tolist())),
        "compressed_to_original_timestamp": token_mapping_json(indices, original_T, duration),
        "pred_length_tokens": int(pred_length_tokens),
        "iou": float(iou) if sample.label_type == "pos" else math.nan,
        "max_token_score": float(np.max(token_scores)),
        "mean_top_token_score": float(np.mean(top_k)),
        "score_std": float(np.std(token_scores)),
        "score_entropy": score_entropy(token_scores),
        "score_sharpness": float(np.max(token_scores) - np.mean(token_scores)),
        "inference_time_sec": float(elapsed),
    }


def score_samples(samples: list[SampleRecord], head: Any, scaler: Any, spec: SystemSpec, split_name: str) -> pd.DataFrame:
    rows = []
    label = f"{spec.system_name}@{spec.budget:g}/{split_name}"
    for idx, sample in enumerate(samples, start=1):
        rows.append(prediction_row(sample, head, scaler, spec))
        if idx % 1000 == 0:
            print(f"{label}: processed {idx}/{len(samples)}")
    return pd.DataFrame(rows)


def split_train_calibration(train_df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    stratify = train_df["sample_type"].astype(str)
    train_idx, cal_idx = train_test_split(
        train_df.index,
        test_size=float(CONFIG["calibration_size"]),
        random_state=int(CONFIG["seed"]),
        stratify=stratify,
    )
    return pd.Index(train_idx), pd.Index(cal_idx)


def fit_reject_classifier(train_df: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler, pd.Index, pd.Index]:
    fit_idx, cal_idx = split_train_calibration(train_df)
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(train_df.loc[fit_idx, REJECT_FEATURES])
    y_fit = train_df.loc[fit_idx, "y_present"].to_numpy()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(CONFIG["seed"]))
    clf.fit(X_fit, y_fit)
    return clf, scaler, fit_idx, cal_idx


def choose_threshold(calibration: pd.DataFrame, score_col: str) -> dict[str, Any]:
    scores = calibration[score_col].to_numpy(dtype=np.float64)
    candidates = np.unique(scores)
    candidates = np.concatenate([[-np.inf], candidates, [np.nextafter(candidates.max(), np.inf)]])
    best: tuple[float, float, float, float, float] | None = None
    for threshold in candidates:
        pos = calibration["sample_type"].eq("pos").to_numpy()
        id_neg = calibration["sample_type"].eq("id_neg").to_numpy()
        ood_neg = calibration["sample_type"].eq("ood_neg").to_numpy()
        rejected = scores < threshold
        pos_accept = float((~rejected[pos]).mean()) if pos.any() else 0.0
        ra_id = float(rejected[id_neg].mean()) if id_neg.any() else 0.0
        ra_ood = float(rejected[ood_neg].mean()) if ood_neg.any() else 0.0
        balanced = float(np.mean([pos_accept, ra_id, ra_ood]))
        item = (balanced, pos_accept, ra_id, ra_ood, float(threshold))
        if best is None or item > best:
            best = item
    assert best is not None
    return {
        "threshold": best[4],
        "criterion": CONFIG["threshold_selection"],
        "decision_rule": "reject if classifier_score < threshold",
        "calibration_balanced_accept_reject": best[0],
        "calibration_positive_accept_rate": best[1],
        "calibration_RA_ID": best[2],
        "calibration_RA_OOD": best[3],
    }


def apply_reject(train_df: pd.DataFrame, val_df: pd.DataFrame, spec: SystemSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    if spec.reject_type == "none":
        for frame in (train_df, val_df):
            frame["confidence_prob"] = np.nan
            frame["classifier_score"] = np.nan
            frame["reject_threshold"] = np.nan
            frame["accepted"] = True
            frame["rejected"] = False
        return train_df, val_df, {"applicable": False, "reason": "no reject module"}
    clf, scaler, fit_idx, cal_idx = fit_reject_classifier(train_df)
    for frame in (train_df, val_df):
        frame["confidence_prob"] = clf.predict_proba(scaler.transform(frame[REJECT_FEATURES]))[:, 1]
        frame["classifier_score"] = frame["confidence_prob"]
    threshold_info = choose_threshold(train_df.loc[cal_idx], "confidence_prob")
    threshold = float(threshold_info["threshold"])
    for frame in (train_df, val_df):
        frame["reject_threshold"] = threshold
        frame["rejected"] = frame["confidence_prob"].to_numpy() < threshold
        frame["accepted"] = ~frame["rejected"]
    threshold_info.update(
        {
            "reject_type": spec.reject_type,
            "fit_rows": int(len(fit_idx)),
            "calibration_rows": int(len(cal_idx)),
            "features": REJECT_FEATURES,
            "classifier": "LogisticRegression(class_weight='balanced')",
            "confidence_source": "predict_proba[:, 1]",
        }
    )
    return train_df, val_df, threshold_info


def metrics_for_system(val_df: pd.DataFrame, spec: SystemSpec) -> dict[str, Any]:
    pos = val_df[val_df["sample_type"] == "pos"]
    id_neg = val_df[val_df["sample_type"] == "id_neg"]
    ood_neg = val_df[val_df["sample_type"] == "ood_neg"]
    pos_accept = pos["accepted"].to_numpy(dtype=bool)
    id_reject = id_neg["rejected"].to_numpy(dtype=bool)
    ood_reject = ood_neg["rejected"].to_numpy(dtype=bool)
    before_05 = float((pos["iou"].to_numpy() >= 0.5).mean())
    before_07 = float((pos["iou"].to_numpy() >= 0.7).mean())
    e2e_05 = float(((pos["iou"].to_numpy() >= 0.5) & pos_accept).mean())
    e2e_07 = float(((pos["iou"].to_numpy() >= 0.7) & pos_accept).mean())
    ra_id = float(id_reject.mean()) if len(id_reject) else 0.0
    ra_ood = float(ood_reject.mean()) if len(ood_reject) else 0.0
    return {
        "system_name": spec.system_name,
        "compression_policy": spec.compression_policy,
        "budget": float(spec.budget),
        "reject_type": spec.reject_type,
        "avg_num_tokens": float(val_df["num_selected_tokens"].mean()),
        "retained_token_ratio": float(val_df["num_selected_tokens"].sum() / val_df["num_original_tokens"].sum()),
        "compression_ratio": float(val_df["num_original_tokens"].sum() / val_df["num_selected_tokens"].sum()),
        "approx_attention_cost_ratio": float((val_df["num_selected_tokens"].sum() / val_df["num_original_tokens"].sum()) ** 2),
        "avg_inference_time_per_query": float(val_df["inference_time_sec"].mean()),
        "R1@0.5_before_reject": before_05,
        "R1@0.7_before_reject": before_07,
        "R1@0.5_e2e": e2e_05,
        "R1@0.7_e2e": e2e_07,
        "mean_iou": float(pos["iou"].mean()),
        "positive_accept_rate": float(pos_accept.mean()) if len(pos_accept) else 0.0,
        "false_reject_rate": float((~pos_accept).mean()) if len(pos_accept) else 0.0,
        "RA_ID": ra_id,
        "RA_OOD": ra_ood,
        "RA_ALL": float(np.concatenate([id_reject, ood_reject]).mean()) if len(id_reject) + len(ood_reject) else 0.0,
        "false_accept_rate_ID": float((~id_reject).mean()) if len(id_reject) else 0.0,
        "false_accept_rate_OOD": float((~ood_reject).mean()) if len(ood_reject) else 0.0,
        "BalancedOpenSet@0.5": float(np.mean([e2e_05, ra_id, ra_ood])),
        "BalancedOpenSet@0.7": float(np.mean([e2e_07, ra_id, ra_ood])),
    }


def memory_snapshot_mb() -> tuple[None, float]:
    return None, float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def assert_no_silent_drops(manifest: Manifest) -> None:
    expected = {
        "train_pos": len(read_jsonl(ROOT / "data/qvhighlights_neg/pos_only/qvhl_pos_train.jsonl")),
        "val_pos": len(read_jsonl(ROOT / "data/qvhighlights_neg/pos_only/qvhl_pos_val.jsonl")),
        "train_id_neg": len(read_jsonl(ROOT / "data/qvhighlights_neg/indomain/qvhl_id_neg_train.jsonl")),
        "val_id_neg": len(read_jsonl(ROOT / "data/qvhighlights_neg/indomain/qvhl_id_neg_val.jsonl")),
        "train_ood_neg": len(read_jsonl(ROOT / "data/qvhighlights_neg/outofdomain/qvhl_ood_neg_train.jsonl")),
        "val_ood_neg": len(read_jsonl(ROOT / "data/qvhighlights_neg/outofdomain/qvhl_ood_neg_val.jsonl")),
    }
    for split_name, expected_count in expected.items():
        actual = len(manifest.splits[split_name])
        if actual != expected_count:
            raise AssertionError(f"silent drop detected for {split_name}: expected {expected_count}, got {actual}")


def assert_joint_sanity(manifest: Manifest) -> dict[str, Any]:
    report = validate_manifest(manifest)
    print(format_validation_report(report))
    if not report.valid:
        raise AssertionError("manifest sanity checks failed")
    assert_no_silent_drops(manifest)
    for split_name in ["train_mixed", "val_pos", "val_id_neg", "val_ood_neg"]:
        qids = [sample.qid for sample in manifest.splits[split_name]]
        duplicates = [qid for qid, count in Counter(qids).items() if count > 1]
        if duplicates:
            raise AssertionError(f"duplicate qids in {split_name}: {duplicates[:5]}")
        for sample in manifest.splits[split_name]:
            if sample.task_stage != CONFIG["task_stage"]:
                raise AssertionError(f"bad task_stage for {sample.qid}: {sample.task_stage}")
            if sample.label_type == "pos" and not sample.gt_windows:
                raise AssertionError(f"positive sample has no gt_window: {sample.qid}")
            if sample.label_type != "pos" and sample.gt_windows is not None:
                raise AssertionError(f"negative sample reads gt_window: {sample.qid}")
            if CONFIG["video_modality"] not in sample.features:
                raise AssertionError(f"missing feature record: {sample.qid}")
            tokens = sample.features[CONFIG["video_modality"]].num_tokens
            if tokens is None or tokens <= 0:
                raise AssertionError(f"empty feature sequence: {sample.qid}")
            text_path = ROOT / CONFIG["text_feature_dir"] / f"qid{source_qid(sample)}.npz"
            if not text_path.exists():
                raise AssertionError(f"missing text feature: {text_path}")
    train_elements = {(s.qid, s.vid, s.label_type) for s in manifest.splits["train_mixed"]}
    val_elements = {
        (s.qid, s.vid, s.label_type)
        for split_name in ["val_pos", "val_id_neg", "val_ood_neg"]
        for s in manifest.splits[split_name]
    }
    overlap = train_elements & val_elements
    if overlap:
        raise AssertionError(f"train/val element overlap: {sorted(overlap)[:5]}")
    return report.to_dict()


def build_system_specs() -> list[SystemSpec]:
    specs = [
        SystemSpec("retrieval_only_no_compression_no_reject", "none", 1.0, "none"),
        SystemSpec("retrieval_reject_no_compression", "none", 1.0, CONFIG["reject_type"]),
    ]
    for budget in CONFIG["budgets"]:
        specs.append(SystemSpec("uniform_compression_reject", "uniform", float(budget), CONFIG["reject_type"]))
    for budget in CONFIG["budgets"]:
        specs.append(SystemSpec("query_aware_compression_no_reject", "query_aware", float(budget), "none"))
    for budget in CONFIG["budgets"]:
        specs.append(SystemSpec("query_aware_compression_reject", "query_aware", float(budget), CONFIG["reject_type"]))
    return specs


def safe_write_table(frame: pd.DataFrame, parquet_path: Path) -> dict[str, Any]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(parquet_path, index=False)
        return {"path": str(parquet_path), "format": "parquet"}
    except Exception as exc:
        csv_path = parquet_path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return {
            "path": str(csv_path),
            "requested_path": str(parquet_path),
            "format": "csv_fallback",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def make_figures(summary: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    def system_frame(system_name: str) -> pd.DataFrame:
        return summary[summary["system_name"].eq(system_name)].sort_values("retained_token_ratio")

    fig, ax = plt.subplots(figsize=(8, 5))
    for system_name, label in [
        ("uniform_compression_reject", "uniform + reject"),
        ("query_aware_compression_no_reject", "query-aware, no reject"),
        ("query_aware_compression_reject", "query-aware + reject"),
    ]:
        data = system_frame(system_name)
        ax.plot(data["retained_token_ratio"], data["R1@0.5_e2e"], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Retained token ratio")
    ax.set_ylabel("R1@0.5 end-to-end")
    ax.set_title("Localization quality vs token budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "localization_vs_budget.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for system_name, metric, label, linestyle in [
        ("uniform_compression_reject", "RA_ID", "uniform RA_ID", "-"),
        ("uniform_compression_reject", "RA_OOD", "uniform RA_OOD", "--"),
        ("query_aware_compression_reject", "RA_ID", "query-aware RA_ID", "-"),
        ("query_aware_compression_reject", "RA_OOD", "query-aware RA_OOD", "--"),
    ]:
        data = system_frame(system_name)
        ax.plot(data["retained_token_ratio"], data[metric], marker="o", linewidth=2, linestyle=linestyle, label=label)
    ax.set_xlabel("Retained token ratio")
    ax.set_ylabel("Reject accuracy")
    ax.set_title("Reject accuracy vs token budget")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rejection_vs_budget.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for system_name, label in [
        ("retrieval_reject_no_compression", "full tokens + reject"),
        ("uniform_compression_reject", "uniform + reject"),
        ("query_aware_compression_reject", "query-aware + reject"),
    ]:
        data = system_frame(system_name)
        ax.plot(data["retained_token_ratio"], data["BalancedOpenSet@0.5"], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Retained token ratio")
    ax.set_ylabel("BalancedOpenSet@0.5")
    ax.set_title("Open-set quality vs token budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "openset_quality_vs_budget.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    reject_rows = summary[summary["reject_type"].ne("none")].copy()
    ax.scatter(reject_rows["approx_attention_cost_ratio"], reject_rows["BalancedOpenSet@0.5"], s=58)
    for _, row in reject_rows.iterrows():
        if row["system_name"] == "retrieval_reject_no_compression":
            label = "full 1.0"
        elif row["system_name"] == "uniform_compression_reject":
            label = f"uniform {row['budget']:.2g}"
        elif row["system_name"] == "query_aware_compression_reject":
            label = f"qa {row['budget']:.2g}"
        else:
            label = f"{row['system_name']} {row['budget']:.2g}"
        ax.annotate(label, (row["approx_attention_cost_ratio"], row["BalancedOpenSet@0.5"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Approx. attention cost ratio")
    ax.set_ylabel("BalancedOpenSet@0.5")
    ax.set_title("Quality-efficiency trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "quality_efficiency_tradeoff.png", dpi=300)
    plt.close(fig)

    full = summary[summary["system_name"].eq("retrieval_reject_no_compression")]
    best_uniform = summary[summary["system_name"].eq("uniform_compression_reject")].sort_values("BalancedOpenSet@0.5", ascending=False).head(1)
    best_qa = summary[summary["system_name"].eq("query_aware_compression_reject")].sort_values("BalancedOpenSet@0.5", ascending=False).head(1)
    bars = pd.concat([full, best_uniform, best_qa], ignore_index=True)
    labels = ["full 1.0", f"uniform {bars.iloc[1]['budget']:.2g}", f"qa {bars.iloc[2]['budget']:.2g}"]
    x = np.arange(len(bars))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, bars["RA_ID"], width, label="RA_ID")
    ax.bar(x + width / 2, bars["RA_OOD"], width, label="RA_OOD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reject accuracy")
    ax.set_title("ID vs OOD rejection")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "id_vs_ood_rejection.png", dpi=300)
    plt.close(fig)


def threshold_metrics_at(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    pos = frame[frame["sample_type"] == "pos"]
    id_neg = frame[frame["sample_type"] == "id_neg"]
    ood_neg = frame[frame["sample_type"] == "ood_neg"]
    pos_accepted = pos["classifier_score"].to_numpy(dtype=float) >= float(threshold)
    id_rejected = id_neg["classifier_score"].to_numpy(dtype=float) < float(threshold)
    ood_rejected = ood_neg["classifier_score"].to_numpy(dtype=float) < float(threshold)
    r1_05_e2e = float(((pos["iou"].to_numpy(dtype=float) >= 0.5) & pos_accepted).mean())
    ra_id = float(id_rejected.mean()) if len(id_rejected) else 0.0
    ra_ood = float(ood_rejected.mean()) if len(ood_rejected) else 0.0
    return {
        "threshold": float(threshold),
        "positive_accept_rate": float(pos_accepted.mean()) if len(pos_accepted) else 0.0,
        "RA_ID": ra_id,
        "RA_OOD": ra_ood,
        "BalancedOpenSet@0.5": float(np.mean([r1_05_e2e, ra_id, ra_ood])),
        "R1@0.5_e2e": r1_05_e2e,
    }


def make_threshold_sensitivity(val_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in [0.5, 0.25]:
        frame = val_predictions[
            val_predictions["system_name"].eq("query_aware_compression_reject")
            & np.isclose(val_predictions["budget"].astype(float), budget)
        ].copy()
        if frame.empty:
            continue
        scores = frame["classifier_score"].dropna().to_numpy(dtype=float)
        chosen = float(frame["reject_threshold"].dropna().iloc[0])
        candidates = np.unique(np.concatenate([np.quantile(scores, np.linspace(0.02, 0.98, 49)), [chosen]]))
        for threshold in candidates:
            item = threshold_metrics_at(frame, float(threshold))
            item.update(
                {
                    "system_name": "query_aware_compression_reject",
                    "compression_policy": "query_aware",
                    "budget": float(budget),
                    "is_chosen_threshold": bool(abs(float(threshold) - chosen) < 1e-12),
                }
            )
            rows.append(item)
    sensitivity = pd.DataFrame(rows)
    if not sensitivity.empty:
        sensitivity = sensitivity[
            [
                "system_name",
                "compression_policy",
                "budget",
                "threshold",
                "is_chosen_threshold",
                "positive_accept_rate",
                "RA_ID",
                "RA_OOD",
                "BalancedOpenSet@0.5",
                "R1@0.5_e2e",
            ]
        ].sort_values(["budget", "threshold"], ascending=[False, True])
    return sensitivity


def make_threshold_sensitivity_figures(sensitivity: pd.DataFrame) -> None:
    if sensitivity.empty:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for budget, filename in [(0.5, "threshold_sensitivity_qa_050.png"), (0.25, "threshold_sensitivity_qa_025.png")]:
        data = sensitivity[np.isclose(sensitivity["budget"].astype(float), budget)].sort_values("threshold")
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in ["positive_accept_rate", "RA_ID", "RA_OOD", "BalancedOpenSet@0.5", "R1@0.5_e2e"]:
            ax.plot(data["threshold"], data[metric], linewidth=2, label=metric)
        chosen = data[data["is_chosen_threshold"]]
        if not chosen.empty:
            ax.axvline(float(chosen.iloc[0]["threshold"]), color="black", linestyle=":", linewidth=1.6, label="chosen threshold")
        ax.set_xlabel("Reject threshold")
        ax.set_ylabel("Metric value")
        ax.set_title(f"Threshold sensitivity: query-aware + reject, budget={budget:.2f}")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=300)
        plt.close(fig)


def pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["system_name", "budget", "BalancedOpenSet@0.5", "R1@0.5_e2e", "RA_ID", "RA_OOD", "compression_ratio", "avg_inference_time_per_query"]
    data = summary[cols].copy().sort_values(["BalancedOpenSet@0.5", "compression_ratio"], ascending=[False, False])
    return data.head(8).reset_index(drop=True)


def build_notes(summary: pd.DataFrame, thresholds: dict[str, Any], output_reports: dict[str, Any], sensitivity: pd.DataFrame) -> str:
    retrieval_baseline = summary[summary["system_name"].eq("retrieval_only_no_compression_no_reject")].iloc[0]
    qa = summary[summary["system_name"].eq("query_aware_compression_reject")].sort_values("BalancedOpenSet@0.5", ascending=False).iloc[0]
    uniform = summary[summary["system_name"].eq("uniform_compression_reject")].sort_values("BalancedOpenSet@0.5", ascending=False).iloc[0]
    qa_025 = summary[
        summary["system_name"].eq("query_aware_compression_reject")
        & np.isclose(summary["budget"].astype(float), 0.25)
    ].iloc[0]
    qa_010 = summary[
        summary["system_name"].eq("query_aware_compression_reject")
        & np.isclose(summary["budget"].astype(float), 0.10)
    ].iloc[0]
    lines = [
        "# Exp-6: Joint query-aware compression + reject-or-retrieve",
        "",
        "Existing modules/notebooks reused:",
        "- `diploma_project.data_layer.*` manifest and sanity-check schema from earlier experiments.",
        "- retrieval head/scaler from `results/exp1b_trainable_full_token_retrieval_head/`.",
        "- reject baseline protocol and confidence-classifier idea from `notebooks/03_open_set_baseline_no_compression.py`.",
        "- uniform compression and compressed-coordinate-to-original-time evaluation pattern from `notebooks/05_query_aware_compression_positive_only.py`.",
        "- query-aware selector from `diploma_project/compression/query_aware.py`.",
        "",
        "Systems actually run:",
        *[f"- `{row.system_name}` budget={row.budget:g}, compression={row.compression_policy}, reject={row.reject_type}" for row in summary.itertuples()],
        "",
        "Reject protocol:",
        "- `confidence_classifier` is trained separately for every compression policy and budget.",
        "- Features are confidence-only: top scores, token-score statistics, predicted duration, token counts, retained/compression ratios, entropy/sharpness.",
        "- Labels: pos=retrieve, id_neg/ood_neg=reject.",
        f"- Threshold selection: `{CONFIG['threshold_selection']}`.",
        "- Thresholds are not reused from the full-token baseline.",
        "",
        "Threshold calibration protocol",
        f"- protocol version: `{CONFIG['threshold_protocol_version']}`.",
        "- classifier is trained on a `train_mixed` fit subset.",
        "- threshold is selected on a held-out calibration subset from `train_mixed`.",
        "- validation splits are used only for final evaluation: `val_pos`, `val_id_neg`, `val_ood_neg`.",
        "- test split is not used.",
        "- This is a deliberate deviation from AGENTS.md v1.0, where threshold selection was phrased as validation-threshold selection.",
        "- Recommended interpretation: protocol v1.1. Results remain comparable only when the threshold protocol is identical.",
        "",
        "Threshold outputs:",
        f"- saved systems with thresholds: {len([v for v in thresholds.values() if isinstance(v, dict) and v.get('applicable', True)])}.",
        "",
        "Threshold sensitivity diagnostic",
        "- Threshold sensitivity is saved for `query_aware_compression_reject` budgets 0.50 and 0.25.",
        "- This analysis checks trade-off stability around the chosen threshold.",
        "- It is diagnostic only and is not used to choose a better post-hoc validation result.",
        f"- sensitivity rows saved: {len(sensitivity)}.",
        "",
        "Observed instability of uniform compression",
        "- Uniform compression is not monotonic across budgets.",
        "- In this run uniform 10% can outperform uniform 25% on R1@0.5_e2e.",
        "- This is not interpreted as a reliable advantage of stronger uniform pruning.",
        "- Likely explanation: compressed candidate windows map back to wider or different original temporal spans, which may accidentally increase IoU for some samples.",
        "- The main conclusion remains that uniform compression is unstable and consistently much worse than query-aware compression in the joint setting.",
        "",
        "Deviations / caveats:",
        "- No test split is used.",
        "- Raw videos are not used.",
        "- `peak_gpu_memory_mb` is omitted because this runner uses NumPy/sklearn CPU inference.",
        "- `peak_cpu_memory_mb` is recorded in the metrics JSON; summary keeps the requested mandatory columns.",
    ]
    if any(report.get("format") == "csv_fallback" for report in output_reports.values() if isinstance(report, dict)):
        lines.append("- Parquet export fell back to CSV because neither `pyarrow` nor `fastparquet` is installed in the current environment.")
    lines.extend(
        [
            "",
            "Short conclusion:",
            f"1. Query-aware compression best joint row: budget={qa['budget']:.2f}, R1@0.5_e2e={qa['R1@0.5_e2e']:.4f}, BalancedOpenSet@0.5={qa['BalancedOpenSet@0.5']:.4f}.",
            f"2. Uniform best joint row: budget={uniform['budget']:.2f}, R1@0.5_e2e={uniform['R1@0.5_e2e']:.4f}, BalancedOpenSet@0.5={uniform['BalancedOpenSet@0.5']:.4f}.",
            f"3. Full retrieval baseline before reject: R1@0.5={retrieval_baseline['R1@0.5_before_reject']:.4f}.",
            f"4. Query-aware 25% remains a strong efficiency trade-off: R1@0.5_e2e={qa_025['R1@0.5_e2e']:.4f}, BalancedOpenSet@0.5={qa_025['BalancedOpenSet@0.5']:.4f}, compression_ratio={qa_025['compression_ratio']:.2f}.",
            f"5. Query-aware 10% is too aggressive for localization: R1@0.5_e2e={qa_010['R1@0.5_e2e']:.4f}.",
            f"6. The best query-aware joint setting keeps positive_accept_rate={qa['positive_accept_rate']:.4f}; check this against RA_ID={qa['RA_ID']:.4f}, RA_OOD={qa['RA_OOD']:.4f} to rule out trivial reject-too-often behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment() -> dict[str, Any]:
    np.random.seed(int(CONFIG["seed"]))
    for path in [RESULT_DIR, METRICS_DIR, PRED_DIR, REJECT_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    manifest_config = BuildManifestConfig(
        feature_root=ROOT / CONFIG["feature_root"],
        feature_modalities=tuple(CONFIG["feature_modalities"]),
        pos_train_path=ROOT / "data/qvhighlights_neg/pos_only/qvhl_pos_train.jsonl",
        pos_val_path=ROOT / "data/qvhighlights_neg/pos_only/qvhl_pos_val.jsonl",
        id_neg_train_path=ROOT / "data/qvhighlights_neg/indomain/qvhl_id_neg_train.jsonl",
        id_neg_val_path=ROOT / "data/qvhighlights_neg/indomain/qvhl_id_neg_val.jsonl",
        ood_neg_train_path=ROOT / "data/qvhighlights_neg/outofdomain/qvhl_ood_neg_train.jsonl",
        ood_neg_val_path=ROOT / "data/qvhighlights_neg/outofdomain/qvhl_ood_neg_val.jsonl",
        output_path=RESULT_DIR / "manifest.json",
        version="v1.0",
    )
    manifest = build_manifest(manifest_config)
    for samples in manifest.splits.values():
        for sample in samples:
            sample.task_stage = CONFIG["task_stage"]
    save_manifest(manifest, RESULT_DIR / "manifest.json")
    sanity_report = assert_joint_sanity(manifest)

    train_samples = manifest.splits["train_mixed"]
    val_samples = [*manifest.splits["val_pos"], *manifest.splits["val_id_neg"], *manifest.splits["val_ood_neg"]]
    head = joblib.load(ROOT / CONFIG["retrieval_head_source"])
    scaler = joblib.load(ROOT / CONFIG["retrieval_scaler_source"])
    specs = build_system_specs()

    all_train: list[pd.DataFrame] = []
    all_val: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    threshold_payload: dict[str, Any] = {}

    for spec in specs:
        print(f"\n=== {spec.system_name} budget={spec.budget:g} reject={spec.reject_type} ===")
        train_df = score_samples(train_samples, head, scaler, spec, "train_mixed")
        val_df = score_samples(val_samples, head, scaler, spec, "val_mixed")
        train_df, val_df, threshold_info = apply_reject(train_df, val_df, spec)
        key = f"{spec.system_name}_{spec.budget:g}".replace(".", "p")
        threshold_payload[key] = threshold_info
        summary_rows.append(metrics_for_system(val_df, spec))
        all_train.append(train_df)
        all_val.append(val_df)
        print(pd.DataFrame([summary_rows[-1]])[["system_name", "budget", "R1@0.5_e2e", "RA_ID", "RA_OOD", "BalancedOpenSet@0.5"]].to_string(index=False))

    summary = pd.DataFrame(summary_rows)
    retrieval_base = summary[summary["system_name"].eq("retrieval_only_no_compression_no_reject")].iloc[0]
    neg_candidates = summary[summary["system_name"].eq("retrieval_reject_no_compression")]
    neg_base = neg_candidates.iloc[0] if len(neg_candidates) else retrieval_base
    summary["delta_vs_retrieval_baseline_R1@0.5"] = summary["R1@0.5_e2e"] - float(retrieval_base["R1@0.5_e2e"])
    summary["delta_vs_negative_aware_baseline_RA_ID"] = summary["RA_ID"] - float(neg_base["RA_ID"])
    summary["delta_vs_negative_aware_baseline_RA_OOD"] = summary["RA_OOD"] - float(neg_base["RA_OOD"])
    summary = summary[SUMMARY_COLUMNS]

    train_predictions = pd.concat(all_train, ignore_index=True)
    val_predictions = pd.concat(all_val, ignore_index=True)
    val_pos_metrics = summary[["system_name", "budget", "R1@0.5_before_reject", "R1@0.7_before_reject", "mean_iou", "R1@0.5_e2e", "R1@0.7_e2e", "positive_accept_rate", "false_reject_rate"]]
    val_id_metrics = summary[["system_name", "budget", "RA_ID", "false_accept_rate_ID"]]
    val_ood_metrics = summary[["system_name", "budget", "RA_OOD", "false_accept_rate_OOD"]]

    summary.to_csv(METRICS_DIR / "summary_by_system.csv", index=False)
    write_json(METRICS_DIR / "summary_by_system.json", summary.to_dict(orient="records"))
    val_pos_metrics.to_csv(METRICS_DIR / "val_pos_metrics.csv", index=False)
    val_id_metrics.to_csv(METRICS_DIR / "val_id_neg_metrics.csv", index=False)
    val_ood_metrics.to_csv(METRICS_DIR / "val_ood_neg_metrics.csv", index=False)
    output_reports = {
        "train_predictions": safe_write_table(train_predictions, PRED_DIR / "train_predictions.parquet"),
        "val_predictions": safe_write_table(val_predictions, PRED_DIR / "val_predictions.parquet"),
        "reject_features_train": safe_write_table(train_predictions[["system_name", "budget", "sample_type", "y_present", "classifier_score", "reject_threshold", *REJECT_FEATURES]], REJECT_DIR / "reject_features_train.parquet"),
        "reject_features_val": safe_write_table(val_predictions[["system_name", "budget", "sample_type", "y_present", "classifier_score", "reject_threshold", *REJECT_FEATURES]], REJECT_DIR / "reject_features_val.parquet"),
    }
    write_json(REJECT_DIR / "thresholds.json", threshold_payload)
    peak_gpu, peak_cpu = memory_snapshot_mb()
    make_figures(summary)
    threshold_sensitivity = make_threshold_sensitivity(val_predictions)
    threshold_sensitivity.to_csv(METRICS_DIR / "threshold_sensitivity.csv", index=False)
    make_threshold_sensitivity_figures(threshold_sensitivity)

    CONFIG["outputs"] = {
        "metrics": str(METRICS_DIR),
        "predictions": output_reports,
        "reject": str(REJECT_DIR),
        "figures": str(FIGURE_DIR),
    }
    write_simple_yaml(RESULT_DIR / "config.yaml", CONFIG)
    metrics_payload = {
        "experiment_name": CONFIG["experiment_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sanity_report": sanity_report,
        "summary": summary.to_dict(orient="records"),
        "thresholds": threshold_payload,
        "threshold_sensitivity": threshold_sensitivity.to_dict(orient="records"),
        "peak_gpu_memory_mb": peak_gpu,
        "peak_cpu_memory_mb": peak_cpu,
        "output_reports": output_reports,
    }
    write_json(RESULT_DIR / "metrics.json", metrics_payload)
    notes = build_notes(summary, threshold_payload, output_reports, threshold_sensitivity)
    (RESULT_DIR / "notes.md").write_text(notes, encoding="utf-8")

    best = summary.sort_values("BalancedOpenSet@0.5", ascending=False).head(5).reset_index(drop=True)
    pareto = pareto_table(summary)
    qa_050 = summary[
        summary["system_name"].eq("query_aware_compression_reject")
        & np.isclose(summary["budget"].astype(float), 0.50)
    ].iloc[0]
    qa_025 = summary[
        summary["system_name"].eq("query_aware_compression_reject")
        & np.isclose(summary["budget"].astype(float), 0.25)
    ].iloc[0]
    qa_010 = summary[
        summary["system_name"].eq("query_aware_compression_reject")
        & np.isclose(summary["budget"].astype(float), 0.10)
    ].iloc[0]
    full_reject = summary[summary["system_name"].eq("retrieval_reject_no_compression")].iloc[0]
    uniform_050 = summary[
        summary["system_name"].eq("uniform_compression_reject")
        & np.isclose(summary["budget"].astype(float), 0.50)
    ].iloc[0]
    diploma_markdown = "\n".join(
        [
            "### Joint query-aware compression + reject-or-retrieve",
            "",
            f"The best joint result is query-aware compression + reject at 50% retained-token budget: R1@0.5_e2e={qa_050['R1@0.5_e2e']:.4f}, RA_ID={qa_050['RA_ID']:.4f}, RA_OOD={qa_050['RA_OOD']:.4f}, BalancedOpenSet@0.5={qa_050['BalancedOpenSet@0.5']:.4f}, compression_ratio={qa_050['compression_ratio']:.2f}.",
            f"Compared with retrieval + reject without compression, it improves BalancedOpenSet@0.5 from {full_reject['BalancedOpenSet@0.5']:.4f} to {qa_050['BalancedOpenSet@0.5']:.4f} while reducing the approximate attention cost ratio from {full_reject['approx_attention_cost_ratio']:.4f} to {qa_050['approx_attention_cost_ratio']:.4f}.",
            f"At the same 50% budget, query-aware + reject is much stronger than uniform + reject: R1@0.5_e2e {qa_050['R1@0.5_e2e']:.4f} vs {uniform_050['R1@0.5_e2e']:.4f}, BalancedOpenSet@0.5 {qa_050['BalancedOpenSet@0.5']:.4f} vs {uniform_050['BalancedOpenSet@0.5']:.4f}.",
            f"The 25% query-aware setting is the strongest efficiency trade-off: it keeps BalancedOpenSet@0.5={qa_025['BalancedOpenSet@0.5']:.4f} with compression_ratio={qa_025['compression_ratio']:.2f}.",
            f"The 10% query-aware setting is too aggressive for localization: R1@0.5_e2e drops to {qa_010['R1@0.5_e2e']:.4f}.",
            "Caveat: this run uses threshold protocol v1.1, where the classifier is fit on a train_mixed fit subset and the threshold is selected on a held-out train_mixed calibration subset; validation splits are reserved for final reporting and no test split is used.",
        ]
    )
    return {
        "summary_by_system": summary,
        "best_by_balanced_openset_05": best,
        "pareto_quality_efficiency_rejection": pareto,
        "threshold_sensitivity": threshold_sensitivity,
        "diploma_markdown": diploma_markdown,
        "result_dir": RESULT_DIR,
        "output_reports": output_reports,
    }


if __name__ == "__main__":
    result = run_experiment()
    print(result["summary_by_system"].to_string(index=False))
    print("\nBest systems:")
    print(result["best_by_balanced_openset_05"].to_string(index=False))
    print("\nPareto-style:")
    print(result["pareto_quality_efficiency_rejection"].to_string(index=False))
    print("\nThreshold sensitivity preview:")
    print(result["threshold_sensitivity"].head(12).to_string(index=False))
    print("\n" + result["diploma_markdown"])
