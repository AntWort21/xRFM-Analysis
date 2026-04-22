"""Starter utilities for xRFM interpretability comparisons.

The assignment asks for one dataset comparing xRFM AGOP-based importance with
PCA loadings, mutual information, and permutation importance.  The dataset and
final trained model are project decisions, so this file provides reusable
building blocks and a reference-importance CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.common.data_io import load_prepared_dataset, project_path
from src.common.preprocess import fit_transform_splits
from src.common.split import apply_split, load_split_indices, make_train_validation_test_split, save_split_indices, split_file_path
from src.common.utils import get_split_settings, load_common_config, write_json


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch/numpy/list values to numpy arrays."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def find_agop_like_arrays(model: Any, max_depth: int = 5) -> list[dict[str, Any]]:
    """Best-effort introspection for AGOP-like square matrices in an xRFM object.

    TODO: Once the team finalizes the xRFM version and trained-model persistence
    approach, replace or narrow this introspection around the exact public
    attributes exposed by that version.
    """

    found: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(obj: Any, path: str, depth: int) -> None:
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))

        try:
            array = to_numpy(obj)
            if array.ndim == 2 and array.shape[0] == array.shape[1] and np.isfinite(array).all():
                found.append(
                    {
                        "path": path,
                        "shape": list(array.shape),
                        "diagonal": np.diag(array).astype(float).tolist(),
                    }
                )
                return
        except Exception:
            pass

        attributes = getattr(obj, "__dict__", {})
        for name, value in attributes.items():
            if name.startswith("_") and "agop" not in name.lower():
                continue
            visit(value, f"{path}.{name}", depth + 1)

        if isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                visit(value, f"{path}[{index}]", depth + 1)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                visit(value, f"{path}[{key!r}]", depth + 1)

    visit(model, "model", 0)
    return found


def reference_importances(X: np.ndarray, y: np.ndarray, task: str, feature_names: list[str]) -> Any:
    """Compute PCA and mutual-information importances for transformed features."""

    try:
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    except ImportError as exc:
        raise ImportError("pandas and scikit-learn are required. Install dependencies from requirements.txt.") from exc

    pca = PCA(n_components=1, random_state=0)
    pca.fit(X)
    pca_loading = np.abs(pca.components_[0])

    if task == "regression":
        mutual_information = mutual_info_regression(X, y, random_state=0)
    else:
        mutual_information = mutual_info_classif(X, y, random_state=0)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "pca_abs_loading_component_1": pca_loading,
            "mutual_information": mutual_information,
        }
    ).sort_values("mutual_information", ascending=False)


def permutation_importance_frame(estimator: Any, X: np.ndarray, y: np.ndarray, feature_names: list[str], scoring: str) -> Any:
    """Compute permutation importance for a fitted estimator."""

    try:
        import pandas as pd
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        raise ImportError("pandas and scikit-learn are required. Install dependencies from requirements.txt.") from exc

    result = permutation_importance(estimator, X, y, n_repeats=5, random_state=0, scoring=scoring)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create starter interpretability reference importances.")
    parser.add_argument("--dataset", required=True, help="Dataset name from configs/datasets.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to configs/common.yaml.")
    parser.add_argument("--common-config", default="configs/common.yaml", help="Shared experiment config path.")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--output", default=None, help="Output CSV path for PCA/MI importances.")
    parser.add_argument("--reuse-split", action="store_true", help="Reuse an existing split JSON if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common_config = load_common_config(args.common_config)
    seed = int(args.seed if args.seed is not None else common_config.get("random_seed", 42))

    dataset_config, prepared = load_prepared_dataset(args.dataset, config_path=args.datasets_config)
    split_path = split_file_path(
        prepared.name,
        seed,
        common_config.get("project", {}).get("splits_dir", "data/splits"),
    )
    if args.reuse_split and split_path.exists():
        split = load_split_indices(split_path)
    else:
        split = make_train_validation_test_split(prepared.y, prepared.task, seed, **get_split_settings(common_config))
        save_split_indices(split, split_path)

    split_data = apply_split(prepared.X, prepared.y, split)
    processed = fit_transform_splits(
        split_data["X_train"],
        split_data["X_validation"],
        split_data["X_test"],
        dataset_config,
        common_config,
    )

    frame = reference_importances(processed.X_train, split_data["y_train"], prepared.task, processed.feature_names)
    output = project_path(args.output or f"results/summary/{prepared.name}__interpretability_reference.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    metadata_path = Path(str(output).replace(".csv", ".json"))
    write_json(
        metadata_path,
        {
            "dataset": prepared.name,
            "seed": seed,
            "split_path": str(split_path),
            "notes": [
                "PCA and mutual information are computed on the shared preprocessed training split.",
                "TODO: train/load xRFM and call find_agop_like_arrays(model) for AGOP diagnostics.",
                "TODO: pass a fitted model to permutation_importance_frame for permutation importance.",
            ],
        },
    )
    print(f"Wrote reference importances to {output}")


if __name__ == "__main__":
    main()

