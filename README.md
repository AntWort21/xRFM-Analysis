# COMP9417 xRFM Experiment Repository

This repository is the shared experiment codebase for comparing xRFM against
XGBoost and Random Forest on tabular datasets.

The main design rule is that anything affecting fairness across models belongs
in `src/common/`. Model-specific training logic stays in `src/models/`.

## Project Structure

```text
data/
  raw/          # original datasets
  processed/    # optional derived datasets
  splits/       # saved train/validation/test split indices
configs/
  common.yaml
  datasets.yaml
  xrfm_grid.yaml
  baseline_grid.yaml
src/
  common/       # shared data, split, preprocessing, metrics, aggregation
  models/       # model-specific runners
  analysis/     # interpretability and subsampling starters
results/
  raw/          # one JSON file per experiment run
  summary/      # aggregated CSV tables
  figures/      # report figures
notebooks/
  sanity_checks.ipynb
```

## Installation

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU-accelerated xRFM, replace `xrfm` in `requirements.txt` with the CUDA
extra recommended for your machine, for example `xrfm[cu12]`.

## Dataset Configuration

Before running real experiments, edit `configs/datasets.yaml`.

Each dataset entry must define:

- `name`: short identifier used by CLI commands
- `path`: CSV, TSV, or Parquet file under `data/raw/`
- `task`: `regression` or `classification`
- `target_column`: supervised target
- `drop_columns`: identifier or leakage columns to remove for every model
- `categorical_columns`: optional explicit categorical features

Leave unknown dataset-specific details marked with `TODO` until confirmed.

The final dataset set must include at least 5 tabular datasets, at least 2
regression tasks, at least 2 classification tasks, one dataset with `n > 10000`,
one with `d > 50`, and one mixed-type dataset.

## Running Experiments

Run one model on one dataset and seed:

```bash
python -m src.models.run_xrfm --dataset DATASET_NAME --seed 42
python -m src.models.run_xgboost --dataset DATASET_NAME --seed 42
python -m src.models.run_random_forest --dataset DATASET_NAME --seed 42
```

All runners use the same shared workflow:

1. load dataset
2. select target
3. drop configured identifier/leakage columns
4. create or reuse train/validation/test split
5. fit preprocessing on train only
6. transform validation and test
7. tune candidates on validation
8. evaluate the selected candidate once on held-out test
9. save metrics, timing, split sizes, and best parameters to `results/raw/`

Aggregate results:

```bash
python -m src.common.aggregate_results --input results/raw --output results/summary/results.csv
```

## Smoke Test With A Synthetic Dataset

This creates a tiny local CSV and verifies the shared pipeline before real
datasets are chosen.

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 200
df = pd.DataFrame({
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
    "group": rng.choice(["a", "b", "c"], size=n),
})
df["target"] = 2 * df["x1"] - 0.5 * df["x2"] + rng.normal(scale=0.1, size=n)
Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/smoke_regression.csv", index=False)
PY
```

Then add this temporary entry to `configs/datasets.yaml`:

```yaml
  - name: smoke_regression
    path: data/raw/smoke_regression.csv
    task: regression
    target_column: target
    drop_columns: []
    categorical_columns:
      - group
    notes: "Temporary smoke-test dataset."
```

Run:

```bash
python -m src.models.run_random_forest --dataset smoke_regression --seed 42
python -m src.common.aggregate_results --input results/raw --output results/summary/results.csv
```

## Analysis Starters

Interpretability reference importances:

```bash
python -m src.analysis.xrfm_interpretability --dataset DATASET_NAME --seed 42
```

Subsampling split manifest for a large dataset:

```bash
python -m src.analysis.subsample --dataset DATASET_NAME --seed 42 --sizes 500,1000,5000,10000
```

The subsampling manifest includes commands that pass explicit split JSON files
to each model runner via `--split-path`.

## Result Schema

Each raw JSON result includes:

- `dataset`, `task`, `model`, `seed`
- `best_params`
- `validation_score`, `validation_score_metric`, `validation_metrics`
- `test_metrics`
- `training_time_seconds`
- `inference_time_per_sample_seconds`
- `train_size`, `validation_size`, `test_size`
- `feature_dim`
- `target_metadata`
- `created_at`

`results/summary/results.csv` flattens these fields for report tables.

## Extension Notes

- Add new shared behavior only under `src/common/`.
- Add new model runners under `src/models/` with the same result schema.
- Keep notebooks for sanity checks and exploration only; core pipeline logic
  should remain in Python modules.
- Dataset-specific target, leakage, and categorical-feature decisions belong in
  `configs/datasets.yaml`, not in model runners.

