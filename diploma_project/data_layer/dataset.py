from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .manifest_builder import FeatureStore
from .schemas import Manifest, SampleRecord

DatasetMode = Literal[
    "positive_only",
    "negative_aware_train",
    "negative_aware_eval",
    "test_inference",
]


@dataclass(slots=True)
class ManifestDataset:
    manifest: Manifest
    mode: DatasetMode
    subset: str | None = None
    load_features: bool = False
    feature_store: FeatureStore = field(init=False, repr=False)
    samples: list[SampleRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.feature_store = FeatureStore(
            self.manifest.feature_root,
            self.manifest.feature_modalities,
        )
        self.samples = self._select_samples()

    def _select_samples(self) -> list[SampleRecord]:
        if self.mode == "positive_only":
            subset = self.subset or "train"
            key = f"{subset}_pos"
            return list(self.manifest.splits[key])
        if self.mode == "negative_aware_train":
            subset = self.subset or "train"
            key = f"{subset}_mixed"
            return list(self.manifest.splits[key])
        if self.mode == "negative_aware_eval":
            subset = self.subset or "val"
            mapping = {
                "pos": f"{subset}_pos",
                "id_neg": f"{subset}_id_neg",
                "ood_neg": f"{subset}_ood_neg",
            }
            selected: list[SampleRecord] = []
            for key in mapping.values():
                selected.extend(self.manifest.splits[key])
            return selected
        if self.mode == "test_inference":
            if self.subset and self.subset in self.manifest.splits:
                return list(self.manifest.splits[self.subset])
            if "test" in self.manifest.splits:
                return list(self.manifest.splits["test"])
            raise KeyError("manifest has no test split")
        raise ValueError(f"unsupported dataset mode: {self.mode}")

    def grouped_eval_splits(self) -> dict[str, list[SampleRecord]]:
        if self.mode != "negative_aware_eval":
            raise ValueError("grouped_eval_splits is only available for negative_aware_eval")
        subset = self.subset or "val"
        return {
            "pos": list(self.manifest.splits[f"{subset}_pos"]),
            "id_neg": list(self.manifest.splits[f"{subset}_id_neg"]),
            "ood_neg": list(self.manifest.splits[f"{subset}_ood_neg"]),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        payload = sample.to_dict()
        if self.load_features:
            payload["video_features"] = {
                name: self.feature_store.load_array(feature["path"])
                for name, feature in payload["features"].items()
            }
        else:
            payload["video_features"] = None
        return payload
