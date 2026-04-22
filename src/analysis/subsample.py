"""Create reproducible subsampling split files for large-dataset experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.common.data_io import load_prepared_dataset, project_path
from src.common.split import SplitIndices, load_split_indices, make_train_validation_test_split, save_split_indices, split_file_path
from src.common.utils import get_split_settings, load_common_config


def parse_sizes(value: str) -> list[int]:
    """Parse comma-separated training sizes."""

    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("At least one train size is required.")
    return sizes


def subsample_train_indices(train_indices: np.ndarray, y: np.ndarray, task: str, size: int, seed: int) -> np.ndarray:
    """Select a reproducible subset of train indices, stratifying when possible."""

    if size >= len(train_indices):
        return np.sort(train_indices)

    if size <= 0:
        raise ValueError("Subsample size must be positive.")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("scikit-learn is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    y_train = y[train_indices]
    stratify = y_train if task == "classification" and _can_stratify_for_size(y_train, size) else None
    selected, _ = train_test_split(
        train_indices,
        train_size=size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.sort(selected)


def _can_stratify_for_size(y: np.ndarray, size: int) -> bool:
    classes, counts = np.unique(y, return_counts=True)
    return bool(len(classes) > 1 and counts.min() >= 2 and size >= len(classes))


def create_subsample_splits(
    dataset_name: str,
    seed: int,
    sizes: list[int],
    common_config_path: str | Path = "configs/common.yaml",
    datasets_config_path: str | Path = "configs/datasets.yaml",
    output_dir: str | Path = "data/splits/subsamples",
) -> list[dict[str, Any]]:
    """Create split JSON files with smaller train subsets and fixed val/test sets."""

    common_config = load_common_config(common_config_path)
    _, prepared = load_prepared_dataset(dataset_name, config_path=datasets_config_path)
    base_split_path = split_file_path(
        prepared.name,
        seed,
        common_config.get("project", {}).get("splits_dir", "data/splits"),
    )
    if base_split_path.exists():
        base_split = load_split_indices(base_split_path)
    else:
        base_split = make_train_validation_test_split(prepared.y, prepared.task, seed, **get_split_settings(common_config))
        save_split_indices(base_split, base_split_path)

    resolved_output_dir = project_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for size in sizes:
        train_subset = subsample_train_indices(base_split.train, prepared.y, prepared.task, size, seed)
        split = SplitIndices(train=train_subset, validation=base_split.validation, test=base_split.test)
        split_path = resolved_output_dir / f"{prepared.name}__seed_{seed}__train_{len(train_subset)}.json"
        save_split_indices(split, split_path)
        manifest.append(
            {
                "dataset": prepared.name,
                "seed": seed,
                "requested_train_size": size,
                "actual_train_size": len(train_subset),
                "validation_size": len(split.validation),
                "test_size": len(split.test),
                "split_path": str(split_path),
                "random_forest_command": f"python -m src.models.run_random_forest --dataset {prepared.name} --seed {seed} --split-path {split_path}",
                "xgboost_command": f"python -m src.models.run_xgboost --dataset {prepared.name} --seed {seed} --split-path {split_path}",
                "xrfm_command": f"python -m src.models.run_xrfm --dataset {prepared.name} --seed {seed} --split-path {split_path}",
            }
        )
    return manifest


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a CSV manifest of generated subsampling splits and runner commands."""

    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["dataset", "seed", "split_path"]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create subsampling split manifests for a large dataset.")
    parser.add_argument("--dataset", required=True, help="Large dataset name from configs/datasets.yaml.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for base and subsample splits.")
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("500,1000,5000,10000"), help="Comma-separated train sizes.")
    parser.add_argument("--common-config", default="configs/common.yaml", help="Shared experiment config path.")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--output-dir", default="data/splits/subsamples", help="Directory for split JSON files.")
    parser.add_argument("--manifest", default=None, help="Output CSV manifest path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = create_subsample_splits(
        dataset_name=args.dataset,
        seed=args.seed,
        sizes=args.sizes,
        common_config_path=args.common_config,
        datasets_config_path=args.datasets_config,
        output_dir=args.output_dir,
    )
    manifest_path = args.manifest or f"results/summary/{args.dataset}__subsample_splits.csv"
    output = write_manifest(rows, manifest_path)
    print(f"Wrote subsampling manifest to {output}")


if __name__ == "__main__":
    main()

