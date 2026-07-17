"""Aggregate per-run JSON outputs into report-friendly CSV tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.common.data_io import project_path


SUMMARY_COLUMNS = [
    "dataset",
    "task",
    "model",
    "seed",
    "validation_score",
    "validation_score_metric",
    "best_params",
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_accuracy",
    "test_auc_roc",
    "training_time_seconds",
    "inference_time_per_sample_seconds",
    "train_size",
    "validation_size",
    "test_size",
    "feature_dim",
    "created_at",
    "result_path",
]


def flatten_result(result: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    """Flatten one raw result record into a CSV row."""

    row: dict[str, Any] = {
        "dataset": result.get("dataset"),
        "task": result.get("task"),
        "model": result.get("model"),
        "seed": result.get("seed"),
        "validation_score": result.get("validation_score"),
        "validation_score_metric": result.get("validation_score_metric"),
        "best_params": json.dumps(result.get("best_params", {}), sort_keys=True),
        "training_time_seconds": result.get("training_time_seconds"),
        "inference_time_per_sample_seconds": result.get("inference_time_per_sample_seconds"),
        "train_size": result.get("train_size"),
        "validation_size": result.get("validation_size"),
        "test_size": result.get("test_size"),
        "feature_dim": result.get("feature_dim"),
        "created_at": result.get("created_at"),
        "result_path": str(result_path),
    }

    for metric_name, value in result.get("validation_metrics", {}).items():
        row[f"validation_{metric_name}"] = value
    for metric_name, value in result.get("test_metrics", {}).items():
        row[f"test_{metric_name}"] = value
    return row


def aggregate_results(input_dir: str | Path, output_path: str | Path) -> Path:
    """Read raw JSON result files and write one summary CSV."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    resolved_input = project_path(input_dir)
    resolved_output = project_path(output_path)
    rows: list[dict[str, Any]] = []

    if resolved_input.exists():
        for result_path in sorted(resolved_input.glob("*.json")):
            with result_path.open("r", encoding="utf-8") as handle:
                rows.append(flatten_result(json.load(handle), result_path))

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        ordered = [column for column in SUMMARY_COLUMNS if column in frame.columns]
        remaining = [column for column in frame.columns if column not in ordered]
        frame = frame[ordered + remaining]

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved_output, index=False)
    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw experiment JSON files into a CSV summary.")
    parser.add_argument("--input", default="results/raw", help="Directory containing raw JSON result files.")
    parser.add_argument("--output", default="results/summary/results.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = aggregate_results(args.input, args.output)
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()

