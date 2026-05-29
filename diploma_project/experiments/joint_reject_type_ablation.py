from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "diploma_project").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root with AGENTS.md and diploma_project/")


ROOT = find_project_root(Path.cwd().resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diploma_project.experiments.joint_compression_reject as joint  # noqa: E402
from diploma_project.data_layer.manifest_builder import BuildManifestConfig, build_manifest, save_manifest  # noqa: E402


RESULT_DIR = ROOT / "results" / "exp6b_joint_reject_type_ablation"
METRICS_DIR = RESULT_DIR / "metrics"
PRED_DIR = RESULT_DIR / "predictions"
REJECT_DIR = RESULT_DIR / "reject"

CONFIG: dict[str, Any] = {
    **joint.CONFIG,
    "experiment_name": "exp6b_joint_reject_type_ablation",
    "protocol": "AGENTS.md v1.1 / Stage D-E / reject-type ablation under fixed joint protocol",
    "split_protocol": "mixed train = pos + id_neg + ood_neg; val = separate pos / id_neg / ood_neg",
    "task_stage": "joint",
    "budgets": [0.5, 0.25, 0.1],
    "reject_types_compared": ["none", "top_score_threshold", "confidence_classifier"],
    "compression_policies_compared": ["none", "uniform", "query_aware"],
    "query_aware_method": "qa_topk_context_diversity",
    "context_radius": 1,
    "reserve_fraction": 0.2,
    "threshold_selection": "calibration_from_train_mixed_max_mean_PosAccept_RA_ID_RA_OOD",
    "threshold_protocol_version": "v1.1_train_calibration",
    "threshold_fit_split": "train_mixed_fit",
    "threshold_calibration_split": "train_mixed_calibration",
    "final_evaluation_splits": ["val_pos", "val_id_neg", "val_ood_neg"],
    "purpose": (
        "Ablates reject-or-retrieve decision type while holding features, splits, "
        "token budgets, threshold calibration, and evaluation protocol fixed."
    ),
}


def bind_joint_globals() -> None:
    joint.CONFIG.clear()
    joint.CONFIG.update(CONFIG)
    joint.RESULT_DIR = RESULT_DIR
    joint.METRICS_DIR = METRICS_DIR
    joint.PRED_DIR = PRED_DIR
    joint.REJECT_DIR = REJECT_DIR
    joint.FIGURE_DIR = RESULT_DIR / "figures"


def build_system_specs() -> list[joint.SystemSpec]:
    specs = [
        joint.SystemSpec("full_no_reject", "none", 1.0, "none"),
        joint.SystemSpec("full_threshold_reject", "none", 1.0, "top_score_threshold"),
        joint.SystemSpec("full_learned_reject", "none", 1.0, "confidence_classifier"),
    ]
    for policy in ["uniform", "query_aware"]:
        for budget in CONFIG["budgets"]:
            specs.extend(
                [
                    joint.SystemSpec(f"{policy}_no_reject", policy, float(budget), "none"),
                    joint.SystemSpec(f"{policy}_threshold_reject", policy, float(budget), "top_score_threshold"),
                    joint.SystemSpec(f"{policy}_learned_reject", policy, float(budget), "confidence_classifier"),
                ]
            )
    return specs


def apply_reject(train_df: pd.DataFrame, val_df: pd.DataFrame, spec: joint.SystemSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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

    if spec.reject_type == "top_score_threshold":
        _, cal_idx = joint.split_train_calibration(train_df)
        for frame in (train_df, val_df):
            frame["confidence_prob"] = np.nan
            frame["classifier_score"] = frame["top_window_score"].to_numpy(dtype=float)
        threshold_info = joint.choose_threshold(train_df.loc[cal_idx], "classifier_score")
        threshold = float(threshold_info["threshold"])
        for frame in (train_df, val_df):
            frame["reject_threshold"] = threshold
            frame["rejected"] = frame["classifier_score"].to_numpy(dtype=float) < threshold
            frame["accepted"] = ~frame["rejected"]
        threshold_info.update(
            {
                "reject_type": spec.reject_type,
                "fit_rows": 0,
                "calibration_rows": int(len(cal_idx)),
                "features": ["top_window_score"],
                "classifier": None,
                "confidence_source": "top_window_score",
                "decision_rule": "reject if top_window_score < threshold",
            }
        )
        return train_df, val_df, threshold_info

    if spec.reject_type == "confidence_classifier":
        return joint.apply_reject(train_df, val_df, spec)

    raise ValueError(f"unknown reject_type: {spec.reject_type}")


def notes(summary: pd.DataFrame, thresholds: dict[str, Any]) -> str:
    best = summary.sort_values("BalancedOpenSet@0.5", ascending=False).head(5)
    lines = [
        "# Exp-6b: Joint reject-type ablation",
        "",
        "Protocol anchor:",
        "- Follows `AGENTS.md` Stage D/E negative-aware joint protocol.",
        "- Uses the same feature files, mixed train split, separate positive / ID-negative / OOD-negative validation splits, and calibration-from-train threshold protocol v1.1.",
        "- Validation splits are reserved for final reporting; no test split is used.",
        "",
        "Ablated factor:",
        "- `reject_type`: `none` vs `top_score_threshold` vs `confidence_classifier`.",
        "- Compression is held to either no compression, uniform compression, or the current query-aware main policy.",
        "- Budgets for compressed systems: 50%, 25%, 10%. Full-token systems use 100%.",
        "",
        "Systems actually run:",
        *[f"- `{row.system_name}` budget={row.budget:g}, compression={row.compression_policy}, reject={row.reject_type}" for row in summary.itertuples()],
        "",
        "Best rows by BalancedOpenSet@0.5:",
        *[
            f"- `{row['system_name']}` budget={row['budget']:g}: BalancedOpenSet@0.5={row['BalancedOpenSet@0.5']:.4f}, "
            f"R1@0.5_e2e={row['R1@0.5_e2e']:.4f}, RA_ID={row['RA_ID']:.4f}, RA_OOD={row['RA_OOD']:.4f}"
            for _, row in best.iterrows()
        ],
        "",
        "Threshold outputs:",
        f"- saved threshold records: {len(thresholds)}.",
    ]
    return "\n".join(lines) + "\n"


def run_experiment() -> dict[str, Any]:
    bind_joint_globals()
    np.random.seed(int(CONFIG["seed"]))
    for path in [RESULT_DIR, METRICS_DIR, PRED_DIR, REJECT_DIR]:
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
        version="v1.1",
    )
    manifest = build_manifest(manifest_config)
    for samples in manifest.splits.values():
        for sample in samples:
            sample.task_stage = CONFIG["task_stage"]
    save_manifest(manifest, RESULT_DIR / "manifest.json")
    sanity_report = joint.assert_joint_sanity(manifest)

    train_samples = manifest.splits["train_mixed"]
    val_samples = [*manifest.splits["val_pos"], *manifest.splits["val_id_neg"], *manifest.splits["val_ood_neg"]]
    head = joblib.load(ROOT / CONFIG["retrieval_head_source"])
    scaler = joblib.load(ROOT / CONFIG["retrieval_scaler_source"])

    all_train: list[pd.DataFrame] = []
    all_val: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    threshold_payload: dict[str, Any] = {}

    for spec in build_system_specs():
        print(f"\n=== {spec.system_name} budget={spec.budget:g} reject={spec.reject_type} ===")
        train_df = joint.score_samples(train_samples, head, scaler, spec, "train_mixed")
        val_df = joint.score_samples(val_samples, head, scaler, spec, "val_all")
        train_df, val_df, threshold_info = apply_reject(train_df, val_df, spec)
        key = f"{spec.system_name}_{spec.budget:g}".replace(".", "p")
        threshold_payload[key] = threshold_info
        summary_rows.append(joint.metrics_for_system(val_df, spec))
        all_train.append(train_df)
        all_val.append(val_df)

    summary = pd.DataFrame(summary_rows)
    retrieval_base = summary[summary["system_name"].eq("full_no_reject")].iloc[0]
    learned_base = summary[summary["system_name"].eq("full_learned_reject")].iloc[0]
    summary["delta_vs_retrieval_baseline_R1@0.5"] = summary["R1@0.5_e2e"] - float(retrieval_base["R1@0.5_e2e"])
    summary["delta_vs_negative_aware_baseline_RA_ID"] = summary["RA_ID"] - float(learned_base["RA_ID"])
    summary["delta_vs_negative_aware_baseline_RA_OOD"] = summary["RA_OOD"] - float(learned_base["RA_OOD"])
    summary = summary[joint.SUMMARY_COLUMNS]

    train_predictions = pd.concat(all_train, ignore_index=True)
    val_predictions = pd.concat(all_val, ignore_index=True)
    output_reports = {
        "train_predictions": joint.safe_write_table(train_predictions, PRED_DIR / "train_predictions.parquet"),
        "val_predictions": joint.safe_write_table(val_predictions, PRED_DIR / "val_predictions.parquet"),
        "reject_features_train": joint.safe_write_table(
            train_predictions[["system_name", "budget", "sample_type", "y_present", "classifier_score", "reject_threshold", *joint.REJECT_FEATURES]],
            REJECT_DIR / "reject_features_train.parquet",
        ),
        "reject_features_val": joint.safe_write_table(
            val_predictions[["system_name", "budget", "sample_type", "y_present", "classifier_score", "reject_threshold", *joint.REJECT_FEATURES]],
            REJECT_DIR / "reject_features_val.parquet",
        ),
    }

    summary.to_csv(METRICS_DIR / "summary_by_system.csv", index=False)
    joint.write_json(METRICS_DIR / "summary_by_system.json", summary.to_dict(orient="records"))
    summary[["system_name", "budget", "reject_type", "R1@0.5_e2e", "R1@0.7_e2e", "positive_accept_rate", "false_reject_rate"]].to_csv(
        METRICS_DIR / "val_pos_metrics.csv", index=False
    )
    summary[["system_name", "budget", "reject_type", "RA_ID", "false_accept_rate_ID"]].to_csv(METRICS_DIR / "val_id_neg_metrics.csv", index=False)
    summary[["system_name", "budget", "reject_type", "RA_OOD", "false_accept_rate_OOD"]].to_csv(METRICS_DIR / "val_ood_neg_metrics.csv", index=False)
    joint.write_json(REJECT_DIR / "thresholds.json", threshold_payload)

    CONFIG["outputs"] = {"metrics": str(METRICS_DIR), "predictions": output_reports, "reject": str(REJECT_DIR)}
    joint.write_simple_yaml(RESULT_DIR / "config.yaml", CONFIG)
    peak_gpu, peak_cpu = joint.memory_snapshot_mb()
    payload = {
        "experiment_name": CONFIG["experiment_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sanity_report": sanity_report,
        "summary": summary.to_dict(orient="records"),
        "thresholds": threshold_payload,
        "peak_gpu_memory_mb": peak_gpu,
        "peak_cpu_memory_mb": peak_cpu,
        "output_reports": output_reports,
    }
    joint.write_json(RESULT_DIR / "metrics.json", payload)
    (RESULT_DIR / "notes.md").write_text(notes(summary, threshold_payload), encoding="utf-8")
    return {"summary_by_system": summary, "result_dir": RESULT_DIR}


if __name__ == "__main__":
    result = run_experiment()
    print(result["summary_by_system"].to_string(index=False))
    print(f"\nSaved to: {result['result_dir']}")
