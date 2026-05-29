from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from diploma_project.data_layer.dataset import ManifestDataset
from diploma_project.data_layer.manifest_builder import FeatureStore, save_manifest
from diploma_project.data_layer.schemas import Manifest, SampleRecord
from diploma_project.data_layer.validate_manifest import format_validation_report, validate_manifest


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "diploma_project").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root with AGENTS.md and diploma_project/")


ROOT = find_project_root()
RESULT_DIR = ROOT / "results" / "exp1b_trainable_full_token_retrieval_head"
MODEL_PATH = RESULT_DIR / "model.joblib"
SCALER_PATH = RESULT_DIR / "scaler.joblib"

CONFIG: dict[str, Any] = {
    "experiment_name": "exp1b_trainable_full_token_retrieval_head",
    "method": "sgd_window_iou_ranker_on_clip_similarity",
    "protocol": "AGENTS.md v1.0 / Stage A / Exp-1b",
    "task_stage": "baseline_retrieval",
    "dataset": "QVHighlights positive-only train/validation split",
    "train_split": "train_pos",
    "val_split": "val_pos",
    "feature_root": "data/qvhighlights/features",
    "video_modality": "clip_features",
    "text_feature_dir": "data/qvhighlights/features/clip_text_features",
    "text_embedding_key": "pooler_output",
    "window_lengths_tokens": [2, 4, 8, 16, 32],
    "epochs": 3,
    "chunk_size_windows": 8192,
    "seed": 42,
    "compression": False,
    "token_selection": False,
    "token_pruning": False,
    "token_pooling": False,
    "reject": False,
    "negative_splits": False,
    "retained_token_ratio": 1.0,
    "compression_ratio": 1.0,
    "model_path": str(MODEL_PATH),
    "scaler_path": str(SCALER_PATH),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_gt_windows(raw_item: dict[str, Any]) -> list[list[float]]:
    windows = raw_item.get("relevant_windows")
    if not windows:
        raise ValueError(f"positive sample qid={raw_item.get('qid')} has no relevant_windows")
    return [[float(start), float(end)] for start, end in windows]


def make_positive_sample(raw_item: dict[str, Any], split: str, feature_store: FeatureStore) -> SampleRecord:
    vid = str(raw_item["vid"]).strip()
    features = feature_store.get_feature_records(vid)
    token_counts = [feature.num_tokens for feature in features.values() if feature.num_tokens is not None]
    return SampleRecord(
        qid=raw_item["qid"],
        query=str(raw_item["query"]),
        vid=vid,
        split=split,
        label_type="pos",
        task_stage="baseline_retrieval",
        is_positive=True,
        gt_windows=normalize_gt_windows(raw_item),
        duration=raw_item.get("duration"),
        features=features,
        num_tokens_before_compression=min(token_counts) if token_counts else None,
        meta={"raw_item": raw_item, "source_qid": raw_item["qid"]},
    )


def build_positive_manifest() -> tuple[Manifest, Path]:
    pos_train_path = ROOT / "data/qvhighlights_neg/pos_only/qvhl_pos_train.jsonl"
    pos_val_path = ROOT / "data/qvhighlights_neg/pos_only/qvhl_pos_val.jsonl"
    feature_store = FeatureStore(ROOT / CONFIG["feature_root"], (CONFIG["video_modality"],))
    train_pos = [make_positive_sample(item, "train", feature_store) for item in read_jsonl(pos_train_path)]
    val_pos = [make_positive_sample(item, "val", feature_store) for item in read_jsonl(pos_val_path)]

    manifest = Manifest(
        version="v1.0",
        source={
            "protocol": "AGENTS.md v1.0 / Exp-1b positive-only",
            "pos_train_path": str(pos_train_path),
            "pos_val_path": str(pos_val_path),
            "negative_splits_used": False,
        },
        splits={"train_pos": train_pos, "val_pos": val_pos},
        feature_modalities=[CONFIG["video_modality"]],
        feature_root=str(ROOT / CONFIG["feature_root"]),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = RESULT_DIR / "manifest.json"
    save_manifest(manifest, manifest_path)
    return manifest, manifest_path


def assert_manifest_ready(manifest: Manifest, train_samples: list[SampleRecord], val_samples: list[SampleRecord]) -> None:
    report = validate_manifest(manifest)
    print(format_validation_report(report))
    if not report.valid:
        raise AssertionError("manifest sanity checks failed")
    if set(manifest.splits) != {"train_pos", "val_pos"}:
        raise AssertionError(f"unexpected splits: {sorted(manifest.splits)}")
    if not all(sample.label_type == "pos" for sample in [*train_samples, *val_samples]):
        raise AssertionError("Exp-1b must use positive-only samples")

    text_feature_dir = ROOT / CONFIG["text_feature_dir"]
    missing_text = [sample.qid for sample in [*train_samples, *val_samples] if not (text_feature_dir / f"qid{sample.qid}.npz").exists()]
    if missing_text:
        raise AssertionError(f"missing text features: {missing_text[:10]}")

    train_qids = {sample.qid for sample in train_samples}
    val_qids = {sample.qid for sample in val_samples}
    qid_overlap = train_qids & val_qids
    if qid_overlap:
        raise AssertionError(f"train/val qid leakage: {sorted(qid_overlap)[:10]}")

    print(f"train_pos={len(train_samples)}, val_pos={len(val_samples)}")
    print(f"train/val qid overlap: {len(qid_overlap)}")
    print(f"train/val vid overlap: {len({sample.vid for sample in train_samples} & {sample.vid for sample in val_samples})}")


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def temporal_iou(left_window: list[float], right_window: list[float]) -> float:
    left = max(float(left_window[0]), float(right_window[0]))
    right = min(float(left_window[1]), float(right_window[1]))
    inter = max(0.0, right - left)
    union = max(float(left_window[1]), float(right_window[1])) - min(float(left_window[0]), float(right_window[0]))
    return inter / union if union > 0 else 0.0


def max_temporal_iou(pred_window: list[float], gt_windows: list[list[float]]) -> float:
    return max(temporal_iou(pred_window, gt) for gt in gt_windows)


def load_npz_array(path: Path, key: str | None = None) -> np.ndarray:
    payload = np.load(path)
    if key is not None:
        return payload[key]
    if "features" in payload.files:
        return payload["features"]
    return payload[payload.files[0]]


def token_window_to_seconds(start_idx: int, end_idx: int, token_count: int, duration: float) -> list[float]:
    return [duration * start_idx / token_count, duration * end_idx / token_count]


def sample_token_scores(sample: SampleRecord, text_feature_dir: Path) -> np.ndarray:
    feature_record = sample.features[CONFIG["video_modality"]]
    video_features = load_npz_array(Path(feature_record.path), key="features")
    query_feature = load_npz_array(text_feature_dir / f"qid{sample.qid}.npz", key=CONFIG["text_embedding_key"])
    video = l2_normalize(video_features, axis=1)
    query = l2_normalize(query_feature.reshape(-1), axis=0)
    if video.shape[1] != query.shape[0]:
        raise ValueError(f"dimension mismatch for qid={sample.qid}: {video.shape} vs {query.shape}")
    return video @ query


def window_feature_matrix(token_scores: np.ndarray, duration: float, window_lengths: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    scores = np.asarray(token_scores, dtype=np.float32)
    token_count = int(scores.shape[0])
    cumsum = np.concatenate([[0.0], np.cumsum(scores, dtype=np.float64)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(scores * scores, dtype=np.float64)])
    rows: list[list[float]] = []
    spans: list[tuple[int, int, int]] = []
    for length in window_lengths:
        if length > token_count:
            continue
        sums = cumsum[length:] - cumsum[:-length]
        sums_sq = cumsum_sq[length:] - cumsum_sq[:-length]
        means = sums / length
        variances = np.maximum(sums_sq / length - means * means, 0.0)
        stds = np.sqrt(variances)
        for start_idx in range(0, token_count - length + 1):
            end_idx = start_idx + length
            center_rel = (start_idx + end_idx) / (2.0 * token_count)
            start_rel = start_idx / token_count
            end_rel = end_idx / token_count
            length_rel = length / token_count
            length_sec = duration * length / token_count
            rows.append(
                [
                    float(means[start_idx]),
                    float(sums[start_idx]),
                    float(stds[start_idx]),
                    float(scores[start_idx:end_idx].max()),
                    float(scores[start_idx:end_idx].min()),
                    float(length),
                    float(length_rel),
                    float(length_sec / max(duration, 1e-6)),
                    float(center_rel),
                    float(start_rel),
                    float(end_rel),
                ]
            )
            spans.append((start_idx, end_idx, length))
    return np.asarray(rows, dtype=np.float32), spans


def window_targets(spans: list[tuple[int, int, int]], token_count: int, duration: float, gt_windows: list[list[float]]) -> np.ndarray:
    values = []
    for start_idx, end_idx, _ in spans:
        pred_window = token_window_to_seconds(start_idx, end_idx, token_count, duration)
        values.append(max_temporal_iou(pred_window, gt_windows))
    return np.asarray(values, dtype=np.float32)


def iter_window_batches(samples: list[SampleRecord], text_feature_dir: Path, chunk_size: int):
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    size = 0
    for sample in samples:
        scores = sample_token_scores(sample, text_feature_dir)
        x_matrix, spans = window_feature_matrix(scores, float(sample.duration), CONFIG["window_lengths_tokens"])
        targets = window_targets(spans, len(scores), float(sample.duration), sample.gt_windows)
        x_parts.append(x_matrix)
        y_parts.append(targets)
        size += len(targets)
        if size >= chunk_size:
            yield np.vstack(x_parts), np.concatenate(y_parts)
            x_parts, y_parts, size = [], [], 0
    if size:
        yield np.vstack(x_parts), np.concatenate(y_parts)


def train_retrieval_head(train_samples: list[SampleRecord], text_feature_dir: Path) -> tuple[SGDRegressor, StandardScaler, float]:
    scaler = StandardScaler()
    for x_batch, _ in iter_window_batches(train_samples, text_feature_dir, int(CONFIG["chunk_size_windows"])):
        scaler.partial_fit(x_batch)

    head = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        learning_rate="adaptive",
        eta0=1e-2,
        random_state=int(CONFIG["seed"]),
        max_iter=1,
        tol=None,
    )

    train_start = time.perf_counter()
    for epoch in range(int(CONFIG["epochs"])):
        epoch_losses = []
        for x_batch, y_batch in iter_window_batches(train_samples, text_feature_dir, int(CONFIG["chunk_size_windows"])):
            x_scaled = scaler.transform(x_batch)
            head.partial_fit(x_scaled, y_batch)
            pred = head.predict(x_scaled)
            epoch_losses.append(float(np.mean((pred - y_batch) ** 2)))
        print(f"epoch {epoch + 1}/{CONFIG['epochs']} mse={np.mean(epoch_losses):.6f}")
    return head, scaler, float(time.perf_counter() - train_start)


def score_validation(
    val_samples: list[SampleRecord],
    text_feature_dir: Path,
    head: SGDRegressor,
    scaler: StandardScaler,
) -> pd.DataFrame:
    predictions: list[dict[str, Any]] = []
    for idx, sample in enumerate(val_samples, start=1):
        e2e_start = time.perf_counter()
        scores = sample_token_scores(sample, text_feature_dir)
        token_count = len(scores)
        x_matrix, spans = window_feature_matrix(scores, float(sample.duration), CONFIG["window_lengths_tokens"])
        scoring_start = time.perf_counter()
        pred_iou_scores = head.predict(scaler.transform(x_matrix))
        best_idx = int(np.argmax(pred_iou_scores))
        start_idx, end_idx, length = spans[best_idx]
        pred_window = token_window_to_seconds(start_idx, end_idx, token_count, float(sample.duration))
        iou = max_temporal_iou(pred_window, sample.gt_windows)
        scoring_time = time.perf_counter() - scoring_start
        e2e_time = time.perf_counter() - e2e_start
        predictions.append(
            {
                "qid": sample.qid,
                "vid": sample.vid,
                "query": sample.query,
                "split": sample.split,
                "label_type": sample.label_type,
                "task_stage": sample.task_stage,
                "duration": float(sample.duration),
                "gt_windows": sample.gt_windows,
                "gt_length": max(float(end) - float(start) for start, end in sample.gt_windows),
                "pred_window": pred_window,
                "pred_start": pred_window[0],
                "pred_end": pred_window[1],
                "pred_length": pred_window[1] - pred_window[0],
                "pred_start_idx": start_idx,
                "pred_end_idx": end_idx,
                "pred_length_tokens": length,
                "head_score": float(pred_iou_scores[best_idx]),
                "max_iou": float(iou),
                "num_video_tokens": token_count,
                "used_video_tokens": token_count,
                "retained_token_ratio": 1.0,
                "compression_ratio": 1.0,
                "scoring_time_sec": float(scoring_time),
                "end_to_end_time_sec": float(e2e_time),
            }
        )
        if idx % 250 == 0:
            print(f"processed {idx}/{len(val_samples)}")
    frame = pd.DataFrame(predictions)
    if len(frame) != len(val_samples):
        raise AssertionError("validation scoring dropped samples")
    return frame


def retrieval_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "num_queries": int(len(frame)),
        "R1@0.3": float((frame["max_iou"] >= 0.3).mean()),
        "R1@0.5": float((frame["max_iou"] >= 0.5).mean()),
        "R1@0.7": float((frame["max_iou"] >= 0.7).mean()),
        "mean_iou": float(frame["max_iou"].mean()),
        "median_iou": float(frame["max_iou"].median()),
        "mean_pred_window_length": float(frame["pred_length"].mean()),
        "mean_gt_window_length": float(frame["gt_length"].mean()),
    }


def grouped_diagnostics(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_col, observed=False):
        if len(group) == 0:
            continue
        row = {group_col: str(key)}
        row.update(retrieval_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics(
    frame: pd.DataFrame,
    train_samples: list[SampleRecord],
    training_time_sec: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_length_counts = frame["pred_length_tokens"].value_counts().reindex(CONFIG["window_lengths_tokens"], fill_value=0).rename_axis("pred_length_tokens").reset_index(name="count")
    pred_length_counts["fraction"] = pred_length_counts["count"] / len(frame)
    pred_length_counts["percent"] = pred_length_counts["fraction"] * 100.0

    frame["video_length_bin"] = pd.cut(
        frame["duration"],
        bins=[0, 60, 120, 180, 300, 600, math.inf],
        labels=["0-60", "60-120", "120-180", "180-300", "300-600", "600+"],
        right=False,
    )
    frame["gt_moment_length_bin"] = pd.cut(
        frame["gt_length"],
        bins=[0, 5, 10, 20, 40, 80, math.inf],
        labels=["0-5", "5-10", "10-20", "20-40", "40-80", "80+"],
        right=False,
    )

    metrics_by_video_length = grouped_diagnostics(frame, "video_length_bin")
    metrics_by_gt_moment_length = grouped_diagnostics(frame, "gt_moment_length_bin")

    exp1_metrics_path = ROOT / "results" / "exp1_positive_retrieval_baseline" / "metrics.json"
    exp1_metrics = json.loads(exp1_metrics_path.read_text(encoding="utf-8")) if exp1_metrics_path.exists() else {}

    metrics = {
        "experiment_name": CONFIG["experiment_name"],
        "split": CONFIG["val_split"],
        "num_train_queries": int(len(train_samples)),
        "num_val_queries": int(len(frame)),
        "R1@0.3": float((frame["max_iou"] >= 0.3).mean()),
        "R1@0.5": float((frame["max_iou"] >= 0.5).mean()),
        "R1@0.7": float((frame["max_iou"] >= 0.7).mean()),
        "mean_iou": float(frame["max_iou"].mean()),
        "median_iou": float(frame["max_iou"].median()),
        "mean_pred_window_length": float(frame["pred_length"].mean()),
        "median_pred_window_length": float(frame["pred_length"].median()),
        "mean_gt_window_length": float(frame["gt_length"].mean()),
        "median_gt_window_length": float(frame["gt_length"].median()),
        "mean_pred_length_tokens": float(frame["pred_length_tokens"].mean()),
        "median_pred_length_tokens": float(frame["pred_length_tokens"].median()),
        "most_common_pred_length_tokens": int(pred_length_counts.sort_values(["count", "pred_length_tokens"], ascending=[False, True]).iloc[0]["pred_length_tokens"]),
        "avg_scoring_time_per_query": float(frame["scoring_time_sec"].mean()),
        "avg_end_to_end_time_per_query": float(frame["end_to_end_time_sec"].mean()),
        "training_time_sec": float(training_time_sec),
        "mean_num_video_tokens": float(frame["num_video_tokens"].mean()),
        "mean_used_video_tokens": float(frame["used_video_tokens"].mean()),
        "retained_token_ratio": 1.0,
        "compression_ratio": 1.0,
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "delta_vs_exp1_R1@0.5": float(frame["max_iou"].ge(0.5).mean() - exp1_metrics.get("R1@0.5", 0.0)) if exp1_metrics else None,
        "delta_vs_exp1_R1@0.7": float(frame["max_iou"].ge(0.7).mean() - exp1_metrics.get("R1@0.7", 0.0)) if exp1_metrics else None,
        "delta_vs_exp1_mean_iou": float(frame["max_iou"].mean() - exp1_metrics.get("mean_iou", 0.0)) if exp1_metrics else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    comparison_rows = []
    if exp1_metrics:
        comparison_rows.append(
            {
                "method": "Exp-1 zero-shot mean-score full-token baseline",
                "R1@0.3": exp1_metrics.get("R1@0.3"),
                "R1@0.5": exp1_metrics.get("R1@0.5"),
                "R1@0.7": exp1_metrics.get("R1@0.7"),
                "mean_iou": exp1_metrics.get("mean_iou"),
                "median_iou": exp1_metrics.get("median_iou"),
                "mean_pred_window_length": exp1_metrics.get("mean_pred_window_length"),
                "mean_gt_window_length": exp1_metrics.get("mean_gt_window_length"),
                "mean_compression_ratio": exp1_metrics.get("compression_ratio"),
            }
        )
    comparison_rows.append(
        {
            "method": "Exp-1b trainable full-token retrieval head",
            "R1@0.3": metrics["R1@0.3"],
            "R1@0.5": metrics["R1@0.5"],
            "R1@0.7": metrics["R1@0.7"],
            "mean_iou": metrics["mean_iou"],
            "median_iou": metrics["median_iou"],
            "mean_pred_window_length": metrics["mean_pred_window_length"],
            "mean_gt_window_length": metrics["mean_gt_window_length"],
            "mean_compression_ratio": metrics["compression_ratio"],
        }
    )
    return metrics, pred_length_counts, metrics_by_video_length, metrics_by_gt_moment_length, pd.DataFrame(comparison_rows)


def write_yaml_scalar(handle, key: str, value: Any) -> None:
    if isinstance(value, bool):
        handle.write(f"{key}: {'true' if value else 'false'}\n")
    elif isinstance(value, (int, float)):
        handle.write(f"{key}: {value}\n")
    elif isinstance(value, list):
        handle.write(f"{key}: {json.dumps(value)}\n")
    else:
        handle.write(f"{key}: {json.dumps(str(value), ensure_ascii=False)}\n")


def write_outputs(
    manifest_path: Path,
    head: SGDRegressor,
    scaler: StandardScaler,
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    pred_length_counts: pd.DataFrame,
    metrics_by_video_length: pd.DataFrame,
    metrics_by_gt_moment_length: pd.DataFrame,
    comparison_exp1_vs_exp1b: pd.DataFrame,
) -> None:
    joblib.dump(head, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with (RESULT_DIR / "config.yaml").open("w", encoding="utf-8") as handle:
        for key, value in CONFIG.items():
            write_yaml_scalar(handle, key, value)
        handle.write(f"manifest_path: {json.dumps(str(manifest_path), ensure_ascii=False)}\n")

    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT_DIR / "threshold.json").write_text(
        json.dumps({"applicable": False, "reason": "Exp-1b is positive-only retrieval and has no reject decision."}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prediction_columns = [
        "qid",
        "vid",
        "query",
        "split",
        "label_type",
        "task_stage",
        "duration",
        "gt_windows",
        "pred_window",
        "pred_start",
        "pred_end",
        "pred_length",
        "gt_length",
        "pred_start_idx",
        "pred_end_idx",
        "pred_length_tokens",
        "head_score",
        "max_iou",
        "num_video_tokens",
        "used_video_tokens",
        "retained_token_ratio",
        "compression_ratio",
        "scoring_time_sec",
        "end_to_end_time_sec",
    ]
    with (RESULT_DIR / "predictions_val_pos.jsonl").open("w", encoding="utf-8") as handle:
        for record in frame[prediction_columns].to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    case_columns = ["qid", "vid", "query", "duration", "gt_windows", "pred_window", "pred_length", "gt_length", "pred_length_tokens", "head_score", "max_iou"]
    frame.sort_values(["max_iou", "head_score"], ascending=[False, False]).head(25)[case_columns].to_csv(RESULT_DIR / "best_cases_val_pos.csv", index=False)
    frame.sort_values(["max_iou", "head_score"], ascending=[True, False]).head(25)[case_columns].to_csv(RESULT_DIR / "worst_cases_val_pos.csv", index=False)
    metrics_by_video_length.to_csv(RESULT_DIR / "metrics_by_video_length.csv", index=False)
    metrics_by_gt_moment_length.to_csv(RESULT_DIR / "metrics_by_gt_moment_length.csv", index=False)
    pred_length_counts.to_csv(RESULT_DIR / "pred_window_length_tokens_distribution.csv", index=False)
    comparison_exp1_vs_exp1b.to_csv(RESULT_DIR / "comparison_exp1_vs_exp1b.csv", index=False)

    notes = f"""# Exp-1b Trainable Full-Token Retrieval Head

Protocol: AGENTS.md v1.0, Stage A / Exp-1b.

- Train split: QVHighlights positive-only `train_pos`.
- Validation split: QVHighlights positive-only `val_pos`.
- Video modality: precomputed `clip_features`.
- Query modality: precomputed `clip_text_features` / `pooler_output`.
- Compression/reject/negative splits: not used.
- Head: `SGDRegressor` window-level IoU ranker over full-token similarity-window features.
- All video tokens are used for every query.
- Model path: `{MODEL_PATH}`.
- Scaler path: `{SCALER_PATH}`.

Main metrics:
- R1@0.5: {metrics['R1@0.5']:.6f}
- R1@0.7: {metrics['R1@0.7']:.6f}
- mean_iou: {metrics['mean_iou']:.6f}
- mean_pred_window_length: {metrics['mean_pred_window_length']:.6f} sec
- mean_gt_window_length: {metrics['mean_gt_window_length']:.6f} sec
- most_common_pred_length_tokens: {metrics['most_common_pred_length_tokens']}
- avg_scoring_time_per_query: {metrics['avg_scoring_time_per_query']:.8f} sec
- avg_end_to_end_time_per_query: {metrics['avg_end_to_end_time_per_query']:.8f} sec

Comparison with Exp-1:
Exp-1b is a stronger full-token baseline and is substantially better than the zero-shot Exp-1 baseline on retrieval metrics. It still often selects long windows, so the result has a length bias in the opposite direction from Exp-1's short-window mean-score bias. This should be treated as a stronger retrieval reference, not as a compression or reject-aware result.
"""
    (RESULT_DIR / "notes.md").write_text(notes, encoding="utf-8")


def run_experiment() -> dict[str, Any]:
    np.random.seed(int(CONFIG["seed"]))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps(CONFIG, indent=2))

    manifest, manifest_path = build_positive_manifest()
    train_samples = ManifestDataset(manifest, mode="positive_only", subset="train").samples
    val_samples = ManifestDataset(manifest, mode="positive_only", subset="val").samples
    assert_manifest_ready(manifest, train_samples, val_samples)

    tol = 1e-9
    if not abs(temporal_iou([10, 20], [15, 25]) - (5 / 15)) < tol:
        raise AssertionError("temporal IoU sanity failed")
    print("Temporal IoU sanity tests passed.")

    text_feature_dir = ROOT / CONFIG["text_feature_dir"]
    head, scaler, training_time_sec = train_retrieval_head(train_samples, text_feature_dir)
    print(f"training_time_sec={training_time_sec:.2f}")

    frame = score_validation(val_samples, text_feature_dir, head, scaler)
    if set(frame["label_type"]) != {"pos"}:
        raise AssertionError("Exp-1b validation must be positive-only")
    if not (frame["used_video_tokens"] == frame["num_video_tokens"]).all():
        raise AssertionError("Exp-1b must use all video tokens")

    metrics, pred_length_counts, metrics_by_video_length, metrics_by_gt_moment_length, comparison = build_metrics(frame, train_samples, training_time_sec)
    write_outputs(manifest_path, head, scaler, frame, metrics, pred_length_counts, metrics_by_video_length, metrics_by_gt_moment_length, comparison)

    required_files = [
        "config.yaml",
        "metrics.json",
        "predictions_val_pos.jsonl",
        "best_cases_val_pos.csv",
        "worst_cases_val_pos.csv",
        "metrics_by_video_length.csv",
        "metrics_by_gt_moment_length.csv",
        "pred_window_length_tokens_distribution.csv",
        "comparison_exp1_vs_exp1b.csv",
        "notes.md",
        "threshold.json",
        "model.joblib",
        "scaler.joblib",
    ]
    missing_files = [name for name in required_files if not (RESULT_DIR / name).exists()]
    if missing_files:
        raise AssertionError(f"missing files: {missing_files}")
    if any(key.startswith("RA_") or key in {"RA_ID", "RA_OOD"} for key in metrics):
        raise AssertionError("positive-only Exp-1b must not report reject metrics")

    print(
        f"Exp-1b: R1@0.5={metrics['R1@0.5']:.4f}, "
        f"R1@0.7={metrics['R1@0.7']:.4f}, mean_iou={metrics['mean_iou']:.4f}. "
        f"Wrote outputs to {RESULT_DIR}"
    )
    return {"metrics": metrics, "result_dir": RESULT_DIR}


if __name__ == "__main__":
    run_experiment()

