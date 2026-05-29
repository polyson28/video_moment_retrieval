from __future__ import annotations

import argparse
import ast
import json
import struct
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import FeatureRecord, Manifest, SampleRecord


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_windows(raw_item: dict[str, Any], label_type: str) -> list[list[float]] | None:
    windows = raw_item.get("relevant_windows")
    if label_type != "pos":
        return None
    if not windows:
        return None
    normalized: list[list[float]] = []
    for start, end in windows:
        normalized.append([float(start), float(end)])
    return normalized


def _infer_stage(split_name: str) -> str:
    if split_name in {"train_pos", "val_pos"}:
        return "baseline_retrieval"
    if "mixed" in split_name:
        return "negative_aware"
    return "negative_aware"


def _parse_npy_shape(npy_bytes: bytes) -> list[int]:
    if npy_bytes[:6] != b"\x93NUMPY":
        raise ValueError("unsupported npy header")
    major = npy_bytes[6]
    if major == 1:
        header_len = struct.unpack("<H", npy_bytes[8:10])[0]
        header_offset = 10
    elif major in {2, 3}:
        header_len = struct.unpack("<I", npy_bytes[8:12])[0]
        header_offset = 12
    else:
        raise ValueError("unsupported npy version")
    header = npy_bytes[header_offset : header_offset + header_len]
    header_dict = ast.literal_eval(header.decode("latin1"))
    return list(header_dict["shape"])


def _read_npz_shape(npz_path: Path) -> list[int] | None:
    with zipfile.ZipFile(npz_path) as archive:
        if "features.npy" not in archive.namelist():
            return None
        with archive.open("features.npy") as handle:
            return _parse_npy_shape(handle.read(256))


@dataclass(slots=True)
class BuildManifestConfig:
    feature_root: Path
    feature_modalities: tuple[str, ...]
    pos_train_path: Path
    pos_val_path: Path
    id_neg_train_path: Path
    id_neg_val_path: Path
    ood_neg_train_path: Path
    ood_neg_val_path: Path
    output_path: Path | None = None
    version: str = "v1.0"

    @classmethod
    def from_local_defaults(cls, output_path: Path | None = None) -> "BuildManifestConfig":
        return cls(
            feature_root=Path("../data/qvhighlights/features"),
            feature_modalities=("clip_features", "slowfast_features"),
            pos_train_path=Path("../data/qvhighlights_neg/pos_only/qvhl_pos_train.jsonl"),
            pos_val_path=Path("../data/qvhighlights_neg/pos_only/qvhl_pos_val.jsonl"),
            id_neg_train_path=Path("../data/qvhighlights_neg/indomain/qvhl_id_neg_train.jsonl"),
            id_neg_val_path=Path("../data/qvhighlights_neg/indomain/qvhl_id_neg_val.jsonl"),
            ood_neg_train_path=Path("../data/qvhighlights_neg/outofdomain/qvhl_ood_neg_train.jsonl"),
            ood_neg_val_path=Path("../data/qvhighlights_neg/outofdomain/qvhl_ood_neg_val.jsonl"),
            output_path=output_path,
        )


class FeatureStore:
    def __init__(self, feature_root: str | Path, modalities: list[str] | tuple[str, ...]):
        self.feature_root = Path(feature_root)
        self.modalities = list(modalities)
        self._index = self._build_index()
        self._record_cache: dict[str, dict[str, FeatureRecord]] = {}

    def _build_index(self) -> dict[str, dict[str, Path]]:
        index: dict[str, dict[str, Path]] = {}
        for modality in self.modalities:
            modality_root = self.feature_root / modality
            modality_index: dict[str, Path] = {}
            if modality_root.exists():
                for path in modality_root.glob("*.npz"):
                    modality_index[path.stem] = path
            index[modality] = modality_index
        return index

    def has_video(self, vid: str) -> bool:
        return any(vid in modality_index for modality_index in self._index.values())

    def get_feature_records(self, vid: str) -> dict[str, FeatureRecord]:
        cached = self._record_cache.get(vid)
        if cached is not None:
            return cached
        records: dict[str, FeatureRecord] = {}
        for modality, modality_index in self._index.items():
            path = modality_index.get(vid)
            if path is None:
                continue
            shape = _read_npz_shape(path)
            num_tokens = shape[0] if shape else None
            records[modality] = FeatureRecord(
                modality=modality,
                path=str(path),
                num_tokens=num_tokens,
                shape=shape,
            )
        self._record_cache[vid] = records
        return records

    def load_array(self, path: str | Path) -> Any:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "numpy is required to load feature arrays; install it or use metadata-only access"
            ) from exc
        payload = np.load(Path(path))
        if "features" in payload.files:
            return payload["features"]
        first_key = payload.files[0]
        return payload[first_key]


def _make_sample(
    raw_item: dict[str, Any],
    split: str,
    label_type: str,
    task_stage: str,
    feature_store: FeatureStore,
) -> SampleRecord:
    vid = str(raw_item["vid"]).strip()
    features = feature_store.get_feature_records(vid)
    source_qid = raw_item["qid"]
    canonical_qid: int | str = source_qid
    if label_type != "pos":
        canonical_qid = f"{label_type}:{source_qid}"
    num_tokens = None
    if features:
        num_tokens = min(
            feature.num_tokens for feature in features.values() if feature.num_tokens is not None
        )
    sample = SampleRecord(
        qid=canonical_qid,
        query=str(raw_item["query"]),
        vid=vid,
        split=split,
        label_type=label_type,
        task_stage=task_stage,
        is_positive=label_type == "pos",
        gt_windows=_normalize_windows(raw_item, label_type),
        duration=raw_item.get("duration"),
        features=features,
        num_tokens_before_compression=num_tokens,
        meta={"raw_item": raw_item, "source_qid": source_qid},
    )
    return sample


def build_manifest(config: BuildManifestConfig) -> Manifest:
    feature_store = FeatureStore(config.feature_root, config.feature_modalities)

    split_sources = {
        "train_pos": (config.pos_train_path, "train", "pos"),
        "val_pos": (config.pos_val_path, "val", "pos"),
        "train_id_neg": (config.id_neg_train_path, "train", "id_neg"),
        "val_id_neg": (config.id_neg_val_path, "val", "id_neg"),
        "train_ood_neg": (config.ood_neg_train_path, "train", "ood_neg"),
        "val_ood_neg": (config.ood_neg_val_path, "val", "ood_neg"),
    }

    split_records: dict[str, list[SampleRecord]] = {}
    for split_name, (path, split, label_type) in split_sources.items():
        items = _read_jsonl(path)
        split_records[split_name] = [
            _make_sample(
                raw_item=item,
                split=split,
                label_type=label_type,
                task_stage=_infer_stage(split_name),
                feature_store=feature_store,
            )
            for item in items
        ]

    split_records["train_mixed"] = [
        *split_records["train_pos"],
        *split_records["train_id_neg"],
        *split_records["train_ood_neg"],
    ]
    split_records["val_mixed"] = [
        *split_records["val_pos"],
        *split_records["val_id_neg"],
        *split_records["val_ood_neg"],
    ]

    manifest = Manifest(
        version=config.version,
        source={
            "protocol": "AGENTS.md v1.0",
            "pos_train_path": str(config.pos_train_path),
            "pos_val_path": str(config.pos_val_path),
            "id_neg_train_path": str(config.id_neg_train_path),
            "id_neg_val_path": str(config.id_neg_val_path),
            "ood_neg_train_path": str(config.ood_neg_train_path),
            "ood_neg_val_path": str(config.ood_neg_val_path),
        },
        splits=split_records,
        feature_modalities=list(config.feature_modalities),
        feature_root=str(config.feature_root),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return manifest


def save_manifest(manifest: Manifest, output_path: str | Path) -> Path:
    return manifest.save(output_path)


def build_manifest_cli() -> None:
    parser = argparse.ArgumentParser(description="Build a unified data manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data_layer/manifest.json"),
        help="Path to the output manifest json.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("../data/qvhighlights/features"),
        help="Root directory containing feature modality folders.",
    )
    parser.add_argument(
        "--feature-modalities",
        nargs="+",
        default=["clip_features", "slowfast_features"],
        help="Feature modality folder names.",
    )
    args = parser.parse_args()

    config = BuildManifestConfig.from_local_defaults(output_path=args.output)
    config = BuildManifestConfig(
        feature_root=args.feature_root,
        feature_modalities=tuple(args.feature_modalities),
        pos_train_path=config.pos_train_path,
        pos_val_path=config.pos_val_path,
        id_neg_train_path=config.id_neg_train_path,
        id_neg_val_path=config.id_neg_val_path,
        ood_neg_train_path=config.ood_neg_train_path,
        ood_neg_val_path=config.ood_neg_val_path,
        output_path=args.output,
        version=config.version,
    )
    manifest = build_manifest(config)
    output_path = save_manifest(manifest, args.output)
    split_sizes = {name: len(items) for name, items in manifest.splits.items()}
    print(json.dumps({"manifest": str(output_path), "split_sizes": split_sizes}, indent=2))


if __name__ == "__main__":
    build_manifest_cli()
