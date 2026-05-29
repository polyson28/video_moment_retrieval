from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .manifest_builder import FeatureStore
from .schemas import Manifest, SampleRecord, ValidationReport


def _check_sample(
    sample: SampleRecord,
    split_name: str,
    feature_store: FeatureStore,
    report: ValidationReport,
) -> None:
    if not feature_store.has_video(sample.vid):
        report.add_issue(
            "error",
            "missing_features",
            f"video {sample.vid} is missing from feature storage",
            split_name=split_name,
            sample_id=sample.qid,
        )
    if sample.label_type == "pos" and not sample.gt_windows:
        report.add_issue(
            "error",
            "missing_gt_windows",
            "positive sample must have non-empty gt_windows",
            split_name=split_name,
            sample_id=sample.qid,
        )
    if sample.label_type != "pos" and sample.gt_windows is not None:
        report.add_issue(
            "error",
            "negative_has_gt",
            "negative sample must have gt_windows=None",
            split_name=split_name,
            sample_id=sample.qid,
        )
    if sample.duration is not None and sample.duration <= 0:
        report.add_issue(
            "error",
            "non_positive_duration",
            "duration must be positive when present",
            split_name=split_name,
            sample_id=sample.qid,
        )
    if not sample.features:
        report.add_issue(
            "error",
            "no_feature_record",
            "sample has no feature records after manifest join",
            split_name=split_name,
            sample_id=sample.qid,
        )
    for feature in sample.features.values():
        if feature.num_tokens is not None and feature.num_tokens <= 0:
            report.add_issue(
                "error",
                "empty_feature_sequence",
                f"feature sequence for modality {feature.modality} is empty",
                split_name=split_name,
                sample_id=sample.qid,
            )


def validate_manifest(manifest: Manifest) -> ValidationReport:
    report = ValidationReport(
        valid=True,
        checked_splits=list(manifest.splits.keys()),
        sample_counts={name: len(items) for name, items in manifest.splits.items()},
    )
    feature_store = FeatureStore(manifest.feature_root, manifest.feature_modalities)

    for split_name, samples in manifest.splits.items():
        qids = [sample.qid for sample in samples]
        duplicates = [qid for qid, count in Counter(qids).items() if count > 1]
        for qid in duplicates:
            report.add_issue(
                "error",
                "duplicate_qid",
                f"qid {qid} is duplicated within split",
                split_name=split_name,
                sample_id=qid,
            )
        for sample in samples:
            _check_sample(sample, split_name, feature_store, report)

    split_groups = {
        "train": {
            sample.qid
            for split_name, samples in manifest.splits.items()
            if split_name.startswith("train_")
            for sample in samples
        },
        "val": {
            sample.qid
            for split_name, samples in manifest.splits.items()
            if split_name.startswith("val_")
            for sample in samples
        },
        "test": {
            sample.qid
            for split_name, samples in manifest.splits.items()
            if split_name.startswith("test")
            for sample in samples
        },
    }
    overlaps = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]
    for left, right in overlaps:
        intersection = split_groups[left] & split_groups[right]
        if intersection:
            preview = sorted(intersection)[:5]
            report.add_issue(
                "error",
                "split_overlap",
                f"{left}/{right} overlap detected for qids: {preview}",
            )

    report.stats = {
        "label_counts": {
            split_name: dict(Counter(sample.label_type for sample in samples))
            for split_name, samples in manifest.splits.items()
        }
    }
    return report


def format_validation_report(report: ValidationReport) -> str:
    lines = [
        f"valid: {report.valid}",
        f"checked_splits: {', '.join(report.checked_splits)}",
    ]
    for split_name, count in report.sample_counts.items():
        lines.append(f"sample_count[{split_name}]: {count}")
    if not report.issues:
        lines.append("issues: none")
        return "\n".join(lines)
    lines.append("issues:")
    for issue in report.issues:
        location = []
        if issue.split_name:
            location.append(f"split={issue.split_name}")
        if issue.sample_id is not None:
            location.append(f"qid={issue.sample_id}")
        suffix = f" ({', '.join(location)})" if location else ""
        lines.append(
            f"- [{issue.severity}] {issue.code}: {issue.message}{suffix}"
        )
    return "\n".join(lines)


def validate_manifest_cli() -> None:
    parser = argparse.ArgumentParser(description="Run manifest sanity checks.")
    parser.add_argument("manifest", type=Path, help="Path to manifest json.")
    args = parser.parse_args()

    manifest = Manifest.load(args.manifest)
    report = validate_manifest(manifest)
    print(format_validation_report(report))
    raise SystemExit(0 if report.valid else 1)


if __name__ == "__main__":
    validate_manifest_cli()
