"""Shared preprocessing fit only on the training split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PreprocessResult:
    """Transformed arrays plus metadata needed by model runners."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    preprocessor: Any
    feature_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    categorical_info: dict[str, Any]


def fit_transform_splits(
    X_train: Any,
    X_validation: Any,
    X_test: Any,
    dataset_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
) -> PreprocessResult:
    """Fit preprocessing on train only and transform validation/test."""

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("scikit-learn is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    preprocessing_config = common_config.get("preprocessing", {})
    categorical_columns = resolve_categorical_columns(X_train, dataset_config, preprocessing_config)
    numeric_columns = [column for column in X_train.columns if column not in set(categorical_columns)]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy=preprocessing_config.get("numeric_imputer", "median"))),
        ]
        if preprocessing_config.get("scale_numeric", True):
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        categorical_steps = [
            ("imputer", SimpleImputer(strategy=preprocessing_config.get("categorical_imputer", "most_frequent"))),
            ("onehot", make_one_hot_encoder()),
        ]
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_columns))

    if not transformers:
        raise ValueError("No feature columns are available for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    X_train_array = to_float32(preprocessor.fit_transform(X_train))
    X_validation_array = to_float32(preprocessor.transform(X_validation))
    X_test_array = to_float32(preprocessor.transform(X_test))

    feature_names = get_feature_names(preprocessor, numeric_columns, categorical_columns)
    categorical_info = build_categorical_info(preprocessor, numeric_columns, categorical_columns)

    return PreprocessResult(
        X_train=X_train_array,
        X_validation=X_validation_array,
        X_test=X_test_array,
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        categorical_info=categorical_info,
    )


def resolve_categorical_columns(
    X_train: Any,
    dataset_config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
) -> list[str]:
    """Use explicit config first; otherwise auto-detect object/category/bool columns."""

    configured = list(dataset_config.get("categorical_columns") or [])
    missing = [column for column in configured if column not in X_train.columns]
    if missing:
        raise ValueError(f"Configured categorical columns are missing: {missing}")
    if configured:
        return configured

    if not preprocessing_config.get("auto_detect_categorical", True):
        return []
    return list(X_train.select_dtypes(include=["object", "category", "bool"]).columns)


def make_one_hot_encoder() -> Any:
    """Create a dense one-hot encoder across supported scikit-learn versions."""

    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_float32(array: Any) -> np.ndarray:
    """Convert dense or sparse transformer output to a float32 array."""

    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float32)


def get_feature_names(preprocessor: Any, numeric_columns: list[str], categorical_columns: list[str]) -> list[str]:
    """Return transformed feature names when scikit-learn exposes them."""

    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names = [f"numeric__{column}" for column in numeric_columns]
        if categorical_columns:
            names.extend(f"categorical__{column}" for column in categorical_columns)
        return names


def build_categorical_info(
    preprocessor: Any,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    """Build xRFM-compatible categorical index metadata without importing torch."""

    categorical_indices: list[list[int]] = []
    categorical_levels: dict[str, list[str]] = {}
    start = len(numeric_columns)

    if categorical_columns:
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        one_hot = categorical_pipeline.named_steps["onehot"]
        for column, categories in zip(categorical_columns, one_hot.categories_):
            length = len(categories)
            categorical_indices.append(list(range(start, start + length)))
            categorical_levels[column] = [str(value) for value in categories]
            start += length

    return {
        "numerical_indices": list(range(len(numeric_columns))),
        "categorical_indices": categorical_indices,
        "categorical_levels": categorical_levels,
    }

