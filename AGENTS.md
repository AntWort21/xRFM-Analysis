{\rtf1\ansi\ansicpg1252\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww30040\viewh16040\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # AGENTS.md\
\
This repository is for a COMP9417 group project on machine learning experiments.\
\
## Project goal\
Build a clean, reproducible experiment repository for comparing:\
- xRFM\
- XGBoost or LightGBM\
- Random Forest\
\
across 5 tabular datasets.\
\
The repository should support:\
- train / validation / test split\
- fair model comparison\
- shared preprocessing\
- validation-based hyperparameter tuning\
- final held-out test evaluation\
- result aggregation\
- report-friendly outputs\
- later extension for interpretability analysis and subsampling experiments\
\
## Team roles\
- Yuan Cao (Travis): owns the xRFM line\
- Adit: owns XGBoost / LightGBM and Random Forest baselines\
- Niko: owns report writing and integration of model details into the report\
\
## Core design principle\
Anything that affects fairness of comparison across models must live in shared modules.\
\
Examples of shared logic:\
- data loading\
- target selection\
- leakage column removal\
- train / validation / test splitting\
- random seed control\
- preprocessing\
- metric computation\
- result schema\
- result aggregation\
\
Model-specific logic must be isolated in separate runner files.\
\
## Repository design rules\
1. Keep the code simple, explicit, and reproducible.\
2. Do not overengineer.\
3. Prefer usable starter implementations over empty placeholders.\
4. Use TODO markers where dataset-specific details are still unknown.\
5. Do not invent dataset-specific column names, labels, or assumptions unless clearly marked as TODO.\
6. Keep interfaces consistent across models wherever possible.\
7. Shared pipeline code must be reusable by xRFM and all baselines.\
8. Results must be saved in a structured format that can later be used for tables and figures in the report.\
9. Code should be easy for two collaborators to extend in parallel.\
10. Add docstrings and concise comments where helpful.\
\
## Shared modules that must exist\
The repository should contain shared modules for:\
- data I/O\
- splitting\
- preprocessing\
- metrics\
- utility functions\
- result aggregation\
\
These shared modules must be written so that all model runners can use them.\
\
## Model-specific modules\
Separate model files should exist for:\
- xRFM\
- XGBoost or LightGBM\
- Random Forest\
\
Do not mix model-specific training code into shared modules.\
\
## Expected experiment workflow\
The codebase should support the following workflow:\
1. Load dataset\
2. Select target column\
3. Remove identifier / leakage columns if specified\
4. Split into train / validation / test\
5. Fit preprocessing only on train\
6. Transform validation and test using the fitted preprocessors\
7. Tune hyperparameters on validation\
8. Evaluate final chosen configuration on held-out test\
9. Save metrics, timing, and best parameters\
10. Aggregate results across datasets and models\
\
## Metrics and timing\
The codebase should be able to support:\
- Regression: RMSE, MAE, R\'b2\
- Classification: Accuracy, AUC-ROC\
- Training time\
- Inference time per sample\
\
## Config files\
The repository should include starter config files for:\
- dataset definitions\
- common settings\
- xRFM hyperparameter grid\
- baseline hyperparameter grid\
\
## Result outputs\
Use structured outputs wherever possible, such as JSON or CSV.\
\
At minimum, results should be easy to aggregate into summary tables containing:\
- dataset\
- task\
- model\
- seed\
- best parameters\
- validation score\
- final test score\
- training time\
- inference time per sample\
- train / validation / test sizes\
- feature dimension after preprocessing\
\
## Interpretability and subsampling\
The repository should leave room for:\
- xRFM interpretability analysis\
- comparison with baseline feature-importance methods\
- subsampling experiments on larger datasets\
\
These can start as starter files with TODO markers.\
\
## Notebook policy\
Notebooks may exist for sanity checks only.\
Core pipeline logic should live in Python source files, not notebooks.\
\
## README expectation\
The repository should include a practical README that explains:\
- project structure\
- installation\
- how to run a single experiment\
- how to extend the codebase\
- where results are stored\
\
## Implementation style\
- Python only\
- Use type hints where reasonable\
- Keep naming clear and consistent\
- Avoid unnecessary abstraction\
- Prefer maintainability over cleverness}