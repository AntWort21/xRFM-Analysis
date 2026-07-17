"""Shared train/validation/test split logic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.data_io import project_path


@dataclass(frozen=True)
class SplitIndices:
    """Integer row positions for train, validation, and test subsets."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "train_size": int(len(self.train)),
            "validation_size": int(len(self.validation)),
            "test_size": int(len(self.test)),
        }


def make_train_validation_test_split(
    y: np.ndarray,
    task: str,
    seed: int,
    train_size: float = 0.60,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    stratify_classification: bool = True,
) -> SplitIndices:
    """Create a reproducible split, stratifying classification tasks when possible."""

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("scikit-learn is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0; got {total:.4f}.")

    indices = np.arange(len(y))
    stratify = y if task == "classification" and stratify_classification and _can_stratify(y) else None

    train_validation_idx, test_idx, y_train_validation, _ = train_test_split(
        indices,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    validation_fraction = validation_size / (train_size + validation_size)
    second_stratify = (
        y_train_validation
        if task == "classification" and stratify_classification and _can_stratify(y_train_validation)
        else None
    )

    train_idx, validation_idx = train_test_split(
        train_validation_idx,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=second_stratify,
    )

    return SplitIndices(
        train=np.sort(train_idx),
        validation=np.sort(validation_idx),
        test=np.sort(test_idx),
    )


def _can_stratify(y: np.ndarray) -> bool:
    """Return true when every class has enough samples for stratified splitting."""

    _, counts = np.unique(y, return_counts=True)
    return bool(len(counts) > 1 and counts.min() >= 2)


def apply_split(X: Any, y: np.ndarray, split: SplitIndices) -> dict[str, Any]:
    """Slice features and targets into train/validation/test subsets."""

    return {
        "X_train": X.iloc[split.train].copy(),
        "X_validation": X.iloc[split.validation].copy(),
        "X_test": X.iloc[split.test].copy(),
        "y_train": y[split.train],
        "y_validation": y[split.validation],
        "y_test": y[split.test],
    }


def split_file_path(dataset_name: str, seed: int, splits_dir: str | Path = "data/splits") -> Path:
    """Return the default split-index file path for a dataset/seed pair."""

    return project_path(splits_dir) / f"{dataset_name}__seed_{seed}.json"


def save_split_indices(split: SplitIndices, path: str | Path) -> Path:
    """Persist split indices as JSON so all models can reuse the exact split."""

    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": split.train.astype(int).tolist(),
        "validation": split.validation.astype(int).tolist(),
        "test": split.test.astype(int).tolist(),
    }
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return resolved


def load_split_indices(path: str | Path) -> SplitIndices:
    """Load split indices written by :func:`save_split_indices`."""

    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitIndices(
        train=np.asarray(payload["train"], dtype=int),
        validation=np.asarray(payload["validation"], dtype=int),
        test=np.asarray(payload["test"], dtype=int),
    )

