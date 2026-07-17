"""Run a Random Forest experiment with the shared pipeline."""

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
    candidate_params,
    get_split_settings,
    load_candidates,
    load_common_config,
    result_file_path,
    set_random_seed,
    write_json,
)


MODEL_NAME = "random_forest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Random Forest experiment.")
    parser.add_argument("--dataset", required=True, help="Dataset name from configs/datasets.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to configs/common.yaml.")
    parser.add_argument("--common-config", default="configs/common.yaml", help="Shared experiment config path.")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--grid-config", default="configs/baseline_grid.yaml", help="Baseline grid config path.")
    parser.add_argument("--output-dir", default=None, help="Raw result output directory.")
    parser.add_argument("--reuse-split", action="store_true", help="Reuse an existing split JSON if present.")
    parser.add_argument("--split-path", default=None, help="Optional explicit split JSON path, useful for subsampling.")
    return parser.parse_args()


def make_model(task: str, params: dict[str, Any], seed: int) -> Any:
    """Construct the sklearn Random Forest estimator for a task."""

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:
        raise ImportError("scikit-learn is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    params = dict(params)
    params.setdefault("random_state", seed)
    params.setdefault("n_jobs", -1)
    if task == "regression":
        return RandomForestRegressor(**params)
    return RandomForestClassifier(**params)


def predict_outputs(model: Any, X: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Predict labels/values and optional probabilities, returning elapsed time."""

    with Timer() as timer:
        probabilities = None
        if task == "classification" and hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X))
            predictions = probabilities
        else:
            predictions = np.asarray(model.predict(X))
    return predictions, probabilities, timer.elapsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    common_config = load_common_config(args.common_config)
    grid_config = load_yaml(args.grid_config)
    seed = int(args.seed if args.seed is not None else common_config.get("random_seed", 42))
    set_random_seed(seed)

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

    candidates = load_candidates(grid_config, MODEL_NAME, prepared.task)
    score_metric = primary_metric_name(prepared.task, common_config)
    best: dict[str, Any] | None = None
    best_score: float | None = None
    tuning_results: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        name = candidate_name(candidate, index)
        params = candidate_params(candidate)
        model = make_model(prepared.task, params, seed)

        with Timer() as fit_timer:
            model.fit(processed.X_train, split_data["y_train"])

        val_predictions, val_probabilities, _ = predict_outputs(model, processed.X_validation, prepared.task)
        val_metrics = compute_metrics(prepared.task, split_data["y_validation"], val_predictions, val_probabilities)
        score = validation_score(val_metrics, prepared.task, common_config)
        tuning_results.append({"name": name, "validation_score": score, "validation_metrics": val_metrics})

        if is_better_score(score, best_score, score_metric):
            best_score = score
            best = {
                "name": name,
                "params": params,
                "model": model,
                "validation_metrics": val_metrics,
                "training_time_seconds": fit_timer.elapsed,
            }

    if best is None or best_score is None:
        raise RuntimeError("No Random Forest candidate was successfully evaluated.")

    test_predictions, test_probabilities, inference_seconds = predict_outputs(best["model"], processed.X_test, prepared.task)
    test_metrics = compute_metrics(prepared.task, split_data["y_test"], test_predictions, test_probabilities)
    inference_time_per_sample = inference_seconds / max(len(split_data["y_test"]), 1)

    record = build_result_record(
        dataset=prepared.name,
        task=prepared.task,
        model=MODEL_NAME,
        seed=seed,
        best_params={"name": best["name"], **best["params"]},
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
            "tuning_results": tuning_results,
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
