"""Run an xRFM experiment with the shared pipeline."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.common.data_io import load_prepared_dataset, load_yaml
from src.common.metrics import compute_metrics, is_better_score, primary_metric_name, validation_score
from src.common.preprocess import fit_transform_splits
from src.common.split import apply_split, load_split_indices, make_train_validation_test_split, save_split_indices, split_file_path
from src.common.utils import (
    Timer,
    build_result_record,
    candidate_name,
    get_split_settings,
    load_candidates,
    load_common_config,
    result_file_path,
    set_random_seed,
    write_json,
)


MODEL_NAME = "xrfm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an xRFM experiment.")
    parser.add_argument("--dataset", required=True, help="Dataset name from configs/datasets.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to configs/common.yaml.")
    parser.add_argument("--common-config", default="configs/common.yaml", help="Shared experiment config path.")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--grid-config", default="configs/xrfm_grid.yaml", help="xRFM grid config path.")
    parser.add_argument("--output-dir", default=None, help="Raw result output directory.")
    parser.add_argument("--device", default="cpu", help="Torch device for xRFM, for example 'cpu' or 'cuda'.")
    parser.add_argument("--reuse-split", action="store_true", help="Reuse an existing split JSON if present.")
    parser.add_argument("--split-path", default=None, help="Optional explicit split JSON path, useful for subsampling.")
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch/numpy/list outputs to a numpy array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def to_torch_float(value: Any, device: Any) -> Any:
    """Convert an array to a float32 torch tensor."""

    import torch

    return torch.as_tensor(value, dtype=torch.float32, device=device)


def xrfm_targets(y: np.ndarray, task: str) -> np.ndarray:
    """Prepare targets using xRFM's expected target dtype convention."""

    if task == "regression":
        return np.asarray(y, dtype=np.float32)
    return np.asarray(y, dtype=np.int64)


def to_torch_target(y: np.ndarray, task: str, device: Any) -> Any:
    """Convert targets for xRFM fit.

    xRFM handles classification label conversion internally when labels are
    integer tensors, including binary and multiclass cases.
    """

    import torch

    dtype = torch.float32 if task == "regression" else torch.long
    return torch.as_tensor(xrfm_targets(y, task), dtype=dtype, device=device)


def torch_categorical_info(categorical_info: dict[str, Any], device: Any) -> dict[str, Any] | None:
    """Convert preprocessing categorical metadata into xRFM's expected tensors."""

    import torch

    categorical_indices = categorical_info.get("categorical_indices", [])
    if not categorical_indices:
        return None

    return {
        "numerical_indices": torch.as_tensor(categorical_info.get("numerical_indices", []), dtype=torch.long, device=device),
        "categorical_indices": [
            torch.as_tensor(indices, dtype=torch.long, device=device) for indices in categorical_indices
        ],
        "categorical_vectors": [
            torch.eye(len(indices), dtype=torch.float32, device=device) for indices in categorical_indices
        ],
    }


def make_model(candidate: dict[str, Any], task: str, categorical_info: dict[str, Any] | None, device: Any) -> Any:
    """Construct the official xRFM estimator from one candidate config."""

    try:
        from xrfm import xRFM
    except ImportError as exc:
        raise ImportError("xrfm is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    tuning_metric = candidate.get("tuning_metric")
    if tuning_metric is None:
        tuning_metric = candidate.get("tuning_metric_by_task", {}).get(task, "mse" if task == "regression" else "accuracy")

    kwargs: dict[str, Any] = {
        "rfm_params": candidate.get("rfm_params", {}),
        "device": device,
        "tuning_metric": tuning_metric,
    }
    if candidate.get("min_subset_size") is not None:
        kwargs["min_subset_size"] = candidate["min_subset_size"]
    if candidate.get("split_method") is not None:
        kwargs["split_method"] = candidate["split_method"]
    if categorical_info is not None:
        kwargs["categorical_info"] = categorical_info
    return xRFM(**kwargs)


def predict_outputs(model: Any, X: Any, task: str) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Predict with xRFM and return optional probabilities plus elapsed time."""

    with Timer() as timer:
        if task == "classification":
            if not hasattr(model, "predict_proba"):
                raise AttributeError("xRFM classification runs require predict_proba for Accuracy/AUC metrics.")
            probabilities = to_numpy(model.predict_proba(X))
            predictions = probabilities
        else:
            probabilities = None
            predictions = to_numpy(model.predict(X))
    return predictions, probabilities, timer.elapsed


def serializable_xrfm_params(candidate: dict[str, Any], name: str) -> dict[str, Any]:
    """Return candidate settings without duplicating metadata."""

    return {
        "name": name,
        "tuning_metric": candidate.get("tuning_metric"),
        "tuning_metric_by_task": candidate.get("tuning_metric_by_task"),
        "min_subset_size": candidate.get("min_subset_size"),
        "split_method": candidate.get("split_method"),
        "rfm_params": candidate.get("rfm_params", {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for xRFM. Install dependencies with `pip install -r requirements.txt`.") from exc

    common_config = load_common_config(args.common_config)
    grid_config = load_yaml(args.grid_config)
    seed = int(args.seed if args.seed is not None else common_config.get("random_seed", 42))
    set_random_seed(seed)

    device = torch.device(args.device)
    dataset_config, prepared = load_prepared_dataset(args.dataset, config_path=args.datasets_config)
    split_settings = get_split_settings(common_config)
    split_path = args.split_path or split_file_path(
        prepared.name,
        seed,
        common_config.get("project", {}).get("splits_dir", "data/splits"),
    )
    if args.split_path:
        split = load_split_indices(split_path)
    elif args.reuse_split and split_path.exists():
        split = load_split_indices(split_path)
    else:
        split = make_train_validation_test_split(prepared.y, prepared.task, seed, **split_settings)
        save_split_indices(split, split_path)

    split_data = apply_split(prepared.X, prepared.y, split)
    processed = fit_transform_splits(
        split_data["X_train"],
        split_data["X_validation"],
        split_data["X_test"],
        dataset_config,
        common_config,
    )

    X_train = to_torch_float(processed.X_train, device)
    X_validation = to_torch_float(processed.X_validation, device)
    X_test = to_torch_float(processed.X_test, device)
    y_train = to_torch_target(split_data["y_train"], prepared.task, device)
    y_validation = to_torch_target(split_data["y_validation"], prepared.task, device)
    categorical_info = torch_categorical_info(processed.categorical_info, device)

    candidates = load_candidates(grid_config)
    score_metric = primary_metric_name(prepared.task, common_config)
    best: dict[str, Any] | None = None
    best_score: float | None = None
    tuning_results: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        name = candidate_name(candidate, index)
        model = make_model(candidate, prepared.task, categorical_info, device)

        with Timer() as fit_timer:
            model.fit(X_train, y_train, X_validation, y_validation)

        val_predictions, val_probabilities, _ = predict_outputs(model, X_validation, prepared.task)
        val_metrics = compute_metrics(prepared.task, split_data["y_validation"], val_predictions, val_probabilities)
        score = validation_score(val_metrics, prepared.task, common_config)
        tuning_results.append({"name": name, "validation_score": score, "validation_metrics": val_metrics})

        if is_better_score(score, best_score, score_metric):
            best_score = score
            best = {
                "name": name,
                "candidate": candidate,
                "model": model,
                "validation_metrics": val_metrics,
                "training_time_seconds": fit_timer.elapsed,
            }

    if best is None or best_score is None:
        raise RuntimeError("No xRFM candidate was successfully evaluated.")

    test_predictions, test_probabilities, inference_seconds = predict_outputs(best["model"], X_test, prepared.task)
    test_metrics = compute_metrics(prepared.task, split_data["y_test"], test_predictions, test_probabilities)
    inference_time_per_sample = inference_seconds / max(len(split_data["y_test"]), 1)

    record = build_result_record(
        dataset=prepared.name,
        task=prepared.task,
        model=MODEL_NAME,
        seed=seed,
        best_params=serializable_xrfm_params(best["candidate"], best["name"]),
        validation_score=best_score,
        validation_score_metric=score_metric,
        validation_metrics=best["validation_metrics"],
        test_metrics=test_metrics,
        training_time_seconds=best["training_time_seconds"],
        inference_time_per_sample_seconds=inference_time_per_sample,
        train_size=len(split_data["y_train"]),
        validation_size=len(split_data["y_validation"]),
        test_size=len(split_data["y_test"]),
        feature_dim=processed.X_train.shape[1],
        target_metadata=prepared.target_metadata,
        extra={
            "target_column": prepared.target_column,
            "split_path": str(split_path),
            "numeric_columns": processed.numeric_columns,
            "categorical_columns": processed.categorical_columns,
            "categorical_info": processed.categorical_info,
            "tuning_results": tuning_results,
            "device": str(device),
        },
    )

    output_dir = args.output_dir or common_config.get("project", {}).get("results_raw_dir", "results/raw")
    output_path = result_file_path(MODEL_NAME, prepared.name, seed, output_dir)
    write_json(output_path, record)
    print(f"Wrote raw result to {output_path}")
    return record


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
