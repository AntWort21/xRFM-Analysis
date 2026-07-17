"""Shared data loading and target handling.

Dataset-specific choices belong in ``configs/datasets.yaml``.  This module
keeps those choices in one place so every model runner uses the same target,
feature set, and leakage-column removals.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_TASKS = {"regression", "classification"}
XLSX_SIGNATURE = b"PK\x03\x04"


@dataclass(frozen=True)
class PreparedDataset:
    """A dataset after shared target selection and column removal."""

    name: str
    task: str
    X: Any
    y: np.ndarray
    feature_columns: list[str]
    target_column: str
    target_metadata: dict[str, Any]


def project_path(path: str | Path) -> Path:
    """Resolve a project-relative path from the repository root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_dataset_config(dataset_name: str, config_path: str | Path = "configs/datasets.yaml") -> dict[str, Any]:
    """Return one dataset entry from the dataset config."""

    config = load_yaml(config_path)
    datasets = config.get("datasets", [])
    for dataset in datasets:
        if dataset.get("name") == dataset_name:
            return dict(dataset)
    known = ", ".join(sorted(str(item.get("name")) for item in datasets))
    raise ValueError(f"Unknown dataset {dataset_name!r}. Known datasets: {known or 'none'}")


def clean_column_name(column: Any) -> str:
    """Normalize source/config column names before lookups.

    Some tabular datasets ship with leading spaces or embedded tabs in headers.
    Cleaning once at the shared loading boundary keeps target/drop lookups fair
    and identical for every model runner.
    """

    return " ".join(str(column).strip().split())


def clean_dataframe_columns(frame: Any) -> Any:
    """Return a dataframe with normalized column names."""

    cleaned_columns = [clean_column_name(column) for column in frame.columns]
    duplicates = sorted(column for column, count in Counter(cleaned_columns).items() if count > 1)
    if duplicates:
        raise ValueError(f"Column cleaning produced duplicate names: {duplicates}")
    if list(frame.columns) == cleaned_columns:
        return frame
    frame = frame.copy()
    frame.columns = cleaned_columns
    return frame


def normalize_dataset_config_columns(dataset_config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize configured column references to match cleaned dataframe headers."""

    normalized = deepcopy(dict(dataset_config))
    if normalized.get("target_column") is not None:
        normalized["target_column"] = clean_column_name(normalized["target_column"])
    normalized["drop_columns"] = [clean_column_name(column) for column in normalized.get("drop_columns") or []]
    normalized["categorical_columns"] = [
        clean_column_name(column) for column in normalized.get("categorical_columns") or []
    ]
    return normalized


def normalize_task(task: str) -> str:
    """Validate and normalize a task string."""

    if not task or str(task).upper().startswith("TODO"):
        raise ValueError("Dataset task is still TODO. Use 'regression' or 'classification'.")
    normalized = str(task).strip().lower()
    if normalized not in VALID_TASKS:
        raise ValueError(f"Unsupported task {task!r}. Expected one of {sorted(VALID_TASKS)}.")
    return normalized


def _require_config_value(config: Mapping[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None or str(value).strip() == "" or str(value).upper().startswith("TODO"):
        name = config.get("name", "<unknown>")
        raise ValueError(f"Dataset {name!r} must define {key!r} before running experiments.")
    return value


def load_dataset_frame(dataset_config: Mapping[str, Any]) -> Any:
    """Load a tabular dataset and normalize its column names."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    path = project_path(_require_config_value(dataset_config, "path"))
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    suffix = path.suffix.lower()
    is_excel_payload = has_xlsx_signature(path)
    if suffix == ".csv":
        frame = pd.read_excel(path) if is_excel_payload else pd.read_csv(path)
        return clean_dataframe_columns(frame)
    if suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
        return clean_dataframe_columns(frame)
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
        return clean_dataframe_columns(frame)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return clean_dataframe_columns(frame)
    raise ValueError(f"Unsupported dataset file type {suffix!r}. Use CSV, TSV, Excel, or Parquet.")


def has_xlsx_signature(path: Path) -> bool:
    """Return true when a file has an XLSX ZIP payload despite its suffix."""

    with path.open("rb") as handle:
        return handle.read(len(XLSX_SIGNATURE)) == XLSX_SIGNATURE


def apply_target_transform(frame: Any, dataset_config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Apply an optional shared target transform before target selection."""

    transform = dataset_config.get("target_transform")
    if transform in {None, ""}:
        return frame, {}
    if transform != "heart_disease_binary":
        raise ValueError(f"Unsupported target_transform {transform!r}.")

    if "num" not in frame.columns:
        raise ValueError("heart_disease_binary target_transform requires source column 'num'.")

    # UCI heart disease stores diagnosis severity in num; the project uses a
    # binary target indicating whether any disease is present.
    transformed = frame.copy()
    transformed["has_disease"] = (transformed["num"].astype(float) > 0).astype(int)
    return transformed, {
        "target_transform": "heart_disease_binary",
        "source_column": "num",
        "rule": "has_disease = 1 if num > 0 else 0",
    }


def prepare_features_target(frame: Any, dataset_config: Mapping[str, Any]) -> PreparedDataset:
    """Apply shared target selection, leakage-column removal, and target encoding."""

    dataset_config = normalize_dataset_config_columns(dataset_config)
    name = str(dataset_config.get("name", "dataset"))
    task = normalize_task(str(dataset_config.get("task", "")))
    frame = clean_dataframe_columns(frame)
    frame, transform_metadata = apply_target_transform(frame, dataset_config)
    target_column = str(_require_config_value(dataset_config, "target_column"))

    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is not present in dataset {name!r}.")

    drop_columns = list(dataset_config.get("drop_columns") or [])
    missing_drop_columns = [column for column in drop_columns if column not in frame.columns]
    if missing_drop_columns:
        raise ValueError(f"Configured drop columns are missing from {name!r}: {missing_drop_columns}")

    y_raw = frame[target_column]
    feature_drop = [target_column, *drop_columns]
    X = frame.drop(columns=feature_drop)
    if X.shape[1] == 0:
        raise ValueError(f"Dataset {name!r} has no feature columns after dropping target/leakage columns.")

    y, target_metadata = encode_target(y_raw, task)
    target_metadata.update(transform_metadata)
    return PreparedDataset(
        name=name,
        task=task,
        X=X,
        y=y,
        feature_columns=list(X.columns),
        target_column=target_column,
        target_metadata=target_metadata,
    )


def encode_target(target: Any, task: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode the target consistently for all model runners."""

    if target.isna().any():
        raise ValueError("Target column contains missing values. Decide how to handle them before running.")

    if task == "regression":
        values = target.astype(float).to_numpy()
        return values, {"encoding": "float"}

    classes = sorted(target.unique().tolist(), key=lambda value: str(value))
    mapping = {value: index for index, value in enumerate(classes)}
    encoded = target.map(mapping).to_numpy(dtype=int)
    metadata = {
        "encoding": "label",
        "classes": [str(value) for value in classes],
        "class_mapping": {str(value): int(index) for value, index in mapping.items()},
    }
    return encoded, metadata


def load_prepared_dataset(
    dataset_name: str,
    config_path: str | Path = "configs/datasets.yaml",
) -> tuple[dict[str, Any], PreparedDataset]:
    """Load one configured dataset and return its config plus prepared data."""

    dataset_config = normalize_dataset_config_columns(get_dataset_config(dataset_name, config_path=config_path))
    frame = load_dataset_frame(dataset_config)
    return dataset_config, prepare_features_target(frame, dataset_config)
