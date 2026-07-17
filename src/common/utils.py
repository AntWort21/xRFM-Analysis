"""General utilities shared across runners."""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.common.data_io import load_yaml, project_path


@dataclass
class Timer:
    """Small context manager for wall-clock timing."""

    start: float = 0.0
    end: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        current_end = self.end or time.perf_counter()
        return float(current_end - self.start)


def set_random_seed(seed: int) -> None:
    """Set seeds for standard libraries used in this project."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_common_config(path: str | Path = "configs/common.yaml") -> dict[str, Any]:
    """Load shared experiment settings."""

    return load_yaml(path)


def get_split_settings(common_config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract split settings with defaults."""

    split_config = common_config.get("split", {})
    return {
        "train_size": float(split_config.get("train_size", 0.60)),
        "validation_size": float(split_config.get("validation_size", 0.20)),
        "test_size": float(split_config.get("test_size", 0.20)),
        "stratify_classification": bool(split_config.get("stratify_classification", True)),
    }


def load_candidates(config: Mapping[str, Any], *keys: str) -> list[dict[str, Any]]:
    """Load explicit candidates or expand a simple parameter grid."""

    section: Mapping[str, Any] = config
    for key in keys:
        section = section[key]

    if "candidates" in section:
        return [copy.deepcopy(candidate) for candidate in section["candidates"]]
    if "grid" in section:
        return expand_grid(section["grid"])
    raise ValueError(f"Config section {'/'.join(keys) or '<root>'} must contain candidates or grid.")


def expand_grid(grid: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a flat hyperparameter grid into candidate dictionaries."""

    keys = list(grid.keys())
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def candidate_name(candidate: Mapping[str, Any], index: int) -> str:
    """Return a readable candidate name."""

    return str(candidate.get("name") or f"candidate_{index}")


def candidate_params(candidate: Mapping[str, Any], exclude: set[str] | None = None) -> dict[str, Any]:
    """Return candidate parameters excluding metadata fields."""

    excluded = {"name"} | (exclude or set())
    return {key: copy.deepcopy(value) for key, value in candidate.items() if key not in excluded}


def now_utc_iso() -> str:
    """Return an ISO timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    """Convert numpy, dataclass, and pathlib values into JSON-safe objects."""

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON file with stable indentation."""

    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def result_file_path(model_name: str, dataset_name: str, seed: int, output_dir: str | Path) -> Path:
    """Return the default raw result path for a model/dataset/seed run."""

    safe_dataset = dataset_name.replace("/", "_").replace(" ", "_")
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    return project_path(output_dir) / f"{safe_dataset}__{safe_model}__seed_{seed}.json"


def build_result_record(
    *,
    dataset: str,
    task: str,
    model: str,
    seed: int,
    best_params: Mapping[str, Any],
    validation_score: float,
    validation_score_metric: str,
    validation_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    training_time_seconds: float,
    inference_time_per_sample_seconds: float,
    train_size: int,
    validation_size: int,
    test_size: int,
    feature_dim: int,
    target_metadata: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared raw-result schema."""

    record: dict[str, Any] = {
        "dataset": dataset,
        "task": task,
        "model": model,
        "seed": int(seed),
        "best_params": dict(best_params),
        "validation_score": float(validation_score),
        "validation_score_metric": validation_score_metric,
        "validation_metrics": dict(validation_metrics),
        "test_metrics": dict(test_metrics),
        "training_time_seconds": float(training_time_seconds),
        "inference_time_per_sample_seconds": float(inference_time_per_sample_seconds),
        "train_size": int(train_size),
        "validation_size": int(validation_size),
        "test_size": int(test_size),
        "feature_dim": int(feature_dim),
        "target_metadata": dict(target_metadata),
        "created_at": now_utc_iso(),
    }
    if extra:
        record.update(dict(extra))
    return record

