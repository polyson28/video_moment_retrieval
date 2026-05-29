from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

LabelType = Literal["pos", "id_neg", "ood_neg"]
DataSplit = Literal["train", "val", "test"]
TaskStage = Literal["baseline_retrieval", "compression", "negative_aware", "joint"]


@dataclass(slots=True)
class FeatureRecord:
    modality: str
    path: str
    num_tokens: int | None = None
    shape: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SampleRecord:
    qid: int | str
    query: str
    vid: str
    split: DataSplit
    label_type: LabelType
    task_stage: TaskStage
    is_positive: bool
    gt_windows: list[list[float]] | None
    duration: float | None
    features: dict[str, FeatureRecord]
    feature_mask: list[int] | None = None
    token_timestamps: list[list[float]] | None = None
    compressible_mask: list[int] | None = None
    num_tokens_before_compression: int | None = None
    relevance_label: int = 0
    negative_subtype: Literal["id", "ood"] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query = " ".join(self.query.split())
        self.vid = str(self.vid).strip()
        self.is_positive = self.label_type == "pos"
        self.relevance_label = int(self.is_positive)
        self.negative_subtype = {
            "id_neg": "id",
            "ood_neg": "ood",
        }.get(self.label_type)

        if self.is_positive:
            if not self.gt_windows:
                raise ValueError(f"positive sample {self.qid} must have gt_windows")
        elif self.gt_windows is not None:
            raise ValueError(f"negative sample {self.qid} must have gt_windows=None")

        if self.duration is not None:
            self.duration = float(self.duration)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = {
            name: feature.to_dict() for name, feature in self.features.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SampleRecord":
        features = {
            name: FeatureRecord(**feature_payload)
            for name, feature_payload in payload["features"].items()
        }
        restored = dict(payload)
        restored["features"] = features
        return cls(**restored)


@dataclass(slots=True)
class Manifest:
    version: str
    source: dict[str, Any]
    splits: dict[str, list[SampleRecord]]
    feature_modalities: list[str]
    feature_root: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "splits": {
                split_name: [sample.to_dict() for sample in samples]
                for split_name, samples in self.splits.items()
            },
            "feature_modalities": self.feature_modalities,
            "feature_root": self.feature_root,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Manifest":
        return cls(
            version=payload["version"],
            source=payload["source"],
            splits={
                split_name: [SampleRecord.from_dict(item) for item in items]
                for split_name, items in payload["splits"].items()
            },
            feature_modalities=list(payload["feature_modalities"]),
            feature_root=payload["feature_root"],
            created_at=payload["created_at"],
        )

    def save(self, output_path: str | Path) -> Path:
        import json

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, manifest_path: str | Path) -> "Manifest":
        import json

        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(slots=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    split_name: str | None = None
    sample_id: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    valid: bool
    checked_splits: list[str]
    sample_counts: dict[str, int]
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        severity: Literal["error", "warning"],
        code: str,
        message: str,
        split_name: str | None = None,
        sample_id: int | str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                split_name=split_name,
                sample_id=sample_id,
            )
        )
        if severity == "error":
            self.valid = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_splits": self.checked_splits,
            "sample_counts": self.sample_counts,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats": self.stats,
        }
