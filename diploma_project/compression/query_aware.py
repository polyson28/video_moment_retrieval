from __future__ import annotations

import math
from typing import Any

import numpy as np


QUERY_AWARE_METHODS = {"qa_topk", "qa_topk_context", "qa_topk_context_diversity"}


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def query_token_similarity(video_clip_tokens: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    video = np.asarray(video_clip_tokens, dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if video.ndim != 2:
        raise ValueError(f"video_clip_tokens must be 2D, got shape={video.shape}")
    if video.shape[0] <= 0:
        raise ValueError("feature length must be > 0")
    if video.shape[1] != query.shape[0]:
        raise ValueError(f"CLIP dimension mismatch: video={video.shape}, query={query.shape}")
    return l2_normalize(video, axis=1) @ l2_normalize(query, axis=0)


def budget_to_k(T: int, budget: float) -> int:
    if T <= 0:
        raise ValueError("feature length must be > 0")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget >= 1.0:
        return T
    return int(min(T, max(1, math.ceil(float(budget) * T))))


def _ranked_indices(scores: np.ndarray) -> list[int]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    # Higher score first, earlier temporal index as deterministic tie-breaker.
    return np.lexsort((np.arange(scores.shape[0]), -scores)).astype(np.int64).tolist()


def _fill_to_budget(selected: list[int], ranked: list[int], k: int) -> list[int]:
    seen = set()
    filled = []
    for idx in selected:
        value = int(idx)
        if value not in seen:
            filled.append(value)
            seen.add(value)
        if len(filled) >= k:
            return filled[:k]
    for idx in ranked:
        value = int(idx)
        if value not in seen:
            filled.append(value)
            seen.add(value)
        if len(filled) >= k:
            break
    return filled


def select_qa_topk(scores: np.ndarray, k: int) -> np.ndarray:
    selected = _ranked_indices(scores)[:k]
    return np.sort(np.asarray(selected, dtype=np.int64))


def select_qa_topk_context(scores: np.ndarray, k: int, context_radius: int = 1) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    T = int(scores.shape[0])
    if context_radius < 0:
        raise ValueError("context_radius must be >= 0")
    if context_radius == 0:
        return select_qa_topk(scores, k)

    ranked = _ranked_indices(scores)
    seed_count = int(min(k, max(1, math.ceil(k / float(2 * context_radius + 1)))))
    seeds = ranked[:seed_count]
    seed_set = set(seeds)
    neighbors = []
    for seed in seeds:
        for offset in range(1, context_radius + 1):
            for candidate in (seed - offset, seed + offset):
                if 0 <= candidate < T and candidate not in seed_set:
                    neighbors.append(candidate)

    unique_neighbors = list(dict.fromkeys(neighbors))
    unique_neighbors.sort(key=lambda idx: (-float(scores[idx]), int(idx)))
    prioritized = [*seeds, *unique_neighbors]
    selected = _fill_to_budget(prioritized, ranked, k)
    return np.sort(np.asarray(selected, dtype=np.int64))


def select_qa_topk_context_diversity(
    scores: np.ndarray,
    k: int,
    context_radius: int = 1,
    reserve_fraction: float = 0.2,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    T = int(scores.shape[0])
    if not 0.0 <= reserve_fraction <= 1.0:
        raise ValueError("reserve_fraction must be in [0, 1]")

    ranked = _ranked_indices(scores)
    if k >= T:
        return np.arange(T, dtype=np.int64)

    k_reserve = int(min(k, math.ceil(float(reserve_fraction) * k)))
    k_query = int(max(0, k - k_reserve))
    query_tokens = (
        select_qa_topk_context(scores, k_query, context_radius).astype(np.int64).tolist()
        if k_query > 0
        else []
    )
    query_set = set(query_tokens)

    reserve_tokens = []
    if k_reserve > 0:
        edges = np.linspace(0, T, num=k_reserve + 1)
        for left, right in zip(edges[:-1], edges[1:]):
            start = int(math.floor(left))
            end = int(math.ceil(right))
            if end <= start:
                end = min(T, start + 1)
            center = int(min(T - 1, max(0, math.floor((start + end - 1) / 2))))
            if center not in query_set:
                reserve_tokens.append(center)

    prioritized = [*query_tokens, *reserve_tokens]
    selected = _fill_to_budget(prioritized, ranked, k)
    return np.sort(np.asarray(selected, dtype=np.int64))


def validate_selected_indices(selected_indices: np.ndarray, T: int, budget: float) -> None:
    indices = np.asarray(selected_indices, dtype=np.int64)
    if T <= 0:
        raise AssertionError("feature length must be > 0")
    if indices.ndim != 1 or len(indices) == 0:
        raise AssertionError("selected_indices must be a non-empty 1D array")
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("selected_indices must be unique")
    if not np.all(indices[:-1] <= indices[1:]):
        raise AssertionError("selected_indices must be sorted by time")
    if int(indices[0]) < 0 or int(indices[-1]) >= T:
        raise AssertionError(f"selected_indices out of range [0, {T - 1}]")
    max_tokens = budget_to_k(T, budget)
    if len(indices) > max_tokens:
        raise AssertionError(f"selected {len(indices)} tokens, budget allows {max_tokens}")
    if budget >= 1.0 and not np.array_equal(indices, np.arange(T, dtype=np.int64)):
        raise AssertionError("budget=1.0 must keep all tokens in original order")


def apply_indices_to_features(
    features: np.ndarray | dict[str, np.ndarray],
    selected_indices: np.ndarray,
    aligned_length: int | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    indices = np.asarray(selected_indices, dtype=np.int64)
    if isinstance(features, np.ndarray):
        return features[indices]
    compressed: dict[str, np.ndarray] = {}
    for name, values in features.items():
        array = np.asarray(values)
        if array.ndim > 0 and (aligned_length is None or int(array.shape[0]) == int(aligned_length)):
            compressed[name] = array[indices]
        else:
            compressed[name] = array
    return compressed


def compress_query_aware(
    features: np.ndarray | dict[str, np.ndarray],
    query_embedding: np.ndarray,
    budget: float,
    method: str,
    context_radius: int = 1,
    reserve_fraction: float = 0.2,
    clip_modality: str = "clip_features",
) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    if method not in QUERY_AWARE_METHODS:
        raise ValueError(f"unknown query-aware compression method: {method}")

    if isinstance(features, dict):
        if clip_modality not in features:
            raise KeyError(f"features dict has no {clip_modality}")
        clip_tokens = np.asarray(features[clip_modality])
    else:
        clip_tokens = np.asarray(features)

    T = int(clip_tokens.shape[0])
    k = budget_to_k(T, budget)
    if budget >= 1.0:
        selected = np.arange(T, dtype=np.int64)
        scores = query_token_similarity(clip_tokens, query_embedding)
    else:
        scores = query_token_similarity(clip_tokens, query_embedding)
        if method == "qa_topk":
            selected = select_qa_topk(scores, k)
        elif method == "qa_topk_context":
            selected = select_qa_topk_context(scores, k, context_radius)
        else:
            selected = select_qa_topk_context_diversity(scores, k, context_radius, reserve_fraction)

    validate_selected_indices(selected, T, budget)
    compressed = apply_indices_to_features(features, selected, aligned_length=T)
    K = int(len(selected))
    metadata = {
        "num_original_tokens": T,
        "num_selected_tokens": K,
        "retained_token_ratio": float(K / T),
        "compression_ratio": float(T / K),
        "method": method,
        "budget": float(budget),
        "context_radius": int(context_radius),
        "reserve_fraction": float(reserve_fraction),
        "selected_indices": selected.astype(int).tolist(),
        "selection_scores": scores.astype(float).tolist(),
    }
    return compressed, selected, metadata
