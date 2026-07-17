# Round-1 vs Round-2 xRFM Tuning Comparison

## Executive Summary

This report compares the first and second rounds of xRFM tuning using the saved experiment outputs in `results/raw/`, `results/raw_round2_xrfm/`, `results/summary/results.csv`, and `results/summary/xrfm_round2_results.csv`. All reported conclusions are based on the saved validation and held-out test metrics for seed 42.

The second-round tuning results are modest and dataset-dependent. The clearest positive evidence is on `online_news_popularity`, where round-two tuning slightly improved validation RMSE, test RMSE, and test R2 relative to round one, although test MAE worsened. On `diabetes_dataset`, round two produced almost no material change in accuracy and did not close the gap to the strongest XGBoost baseline. On `student_dropout_academic_success`, round two improved held-out test accuracy, but validation accuracy was unchanged and AUC-ROC decreased, so the evidence is mixed. On `winequality_white` and `heart_disease_uci`, round-two tuning did not improve the primary metrics.

Overall, these results do not support a claim that xRFM is uniformly superior across the five datasets. They instead suggest that xRFM performance is sensitive to dataset characteristics and tuning choices, with the strongest project evidence appearing on the large regression dataset `online_news_popularity`.

## Tuning Methodology

### Round 1

Round one used a small starter xRFM grid intended to establish a fair baseline without excessive CPU cost. The grid contained three candidates:

- `xrfm_random_pca_light`: a lighter candidate using `random_pca`, `min_subset_size=5000`, `bandwidth=5.0`, `reg=0.001`, and `iters=1`.
- `xrfm_l2_small`: an AGOP-based candidate using `top_vector_agop_on_subset`, `min_subset_size=5000`, `bandwidth=5.0`, `reg=0.001`, and `iters=3`.
- `xrfm_l2_medium`: a stronger AGOP-based candidate using `top_vector_agop_on_subset`, `min_subset_size=10000`, `bandwidth=10.0`, `reg=0.001`, and `iters=5`.

Model selection was performed on the validation split using the configured primary metric: RMSE for regression and accuracy for classification. The selected candidate was then evaluated once on the held-out test split.

### Round 2

Round two changed the grid to a compact AGOP-only tuning set. The revised grid was designed to test a small number of targeted hypotheses rather than retuning every dimension exhaustively:

- whether fewer iterations could reduce cost while preserving performance;
- whether the round-one anchor setting remained competitive;
- whether narrower bandwidth (`bandwidth=3.0`) helped classification datasets;
- whether stronger regularization (`reg=0.01`) helped the large regression dataset;
- whether increasing `min_subset_size` to 10000 was worth the additional CPU cost.

The `random_pca` candidate was removed because it was not competitive in the round-one xRFM tuning results. Round two was originally motivated by the highest-value datasets, especially `online_news_popularity` and `diabetes_dataset`, but the available round-two outputs now cover all five datasets. This report therefore includes all five datasets, while interpreting the smaller-dataset results as diagnostic evidence rather than as the main motivation for the second round.

## Round-1 vs Round-2 Comparison

| Dataset | Task | Round-1 best xRFM | Round-2 best xRFM | Key metric changes | Conclusion changed? |
|---|---|---|---|---|---|
| `winequality_white` | Regression | `xrfm_l2_medium` | `xrfm_agop_5k_bw5_reg01_iter3` | Validation RMSE `0.5976 -> 0.6021`; test RMSE `0.6029 -> 0.6078`; test R2 `0.5307 -> 0.5229` | No. Round 2 was slightly worse; xRFM remains close to Random Forest but not clearly better. |
| `heart_disease_uci` | Classification | `xrfm_l2_medium` | `xrfm_agop_5k_bw5_reg001_iter2` | Validation accuracy `0.8043 -> 0.7989`; test accuracy `0.8587 -> 0.8533`; test AUC `0.8946 -> 0.8959` | No. Accuracy worsened slightly; AUC changed minimally. |
| `diabetes_dataset` | Classification | `xrfm_l2_small` | `xrfm_agop_5k_bw3_reg001_iter3` | Validation accuracy `0.9698 -> 0.9699`; test accuracy `0.9700 -> 0.9700`; test AUC `0.9686 -> 0.9696` | No. Round 2 did not close the XGBoost accuracy gap. |
| `student_dropout_academic_success` | Classification | `xrfm_l2_medium` | `xrfm_agop_5k_bw3_reg001_iter3` | Validation accuracy `0.7831 -> 0.7831`; test accuracy `0.7571 -> 0.7695`; test AUC `0.8718 -> 0.8692` | Partially. Test accuracy improved enough to match/slightly exceed the best baseline test accuracy, but validation accuracy and AUC do not support a strong superiority claim. |
| `online_news_popularity` | Regression | `xrfm_l2_small` | `xrfm_agop_5k_bw5_reg01_iter3` | Validation RMSE `14092.9 -> 14075.8`; test RMSE `10771.7 -> 10750.5`; test MAE `2757.9 -> 2870.0`; test R2 `0.0385 -> 0.0423` | Mostly no, but strengthened. xRFM already had the best test RMSE in round 1 and improved slightly in round 2, though MAE worsened. |

## Results Interpretation

The round-two grid did not produce a uniform improvement across datasets. This is important for the report narrative: xRFM appears to be dataset-dependent in these experiments, rather than consistently dominating the tree-based baselines.

For `online_news_popularity`, stronger regularization produced the selected round-two xRFM model. This improved validation RMSE and test RMSE, and it also slightly increased test R2. However, test MAE worsened, so the improvement should be described as a modest strengthening of the RMSE-based result rather than broad improvement across all regression metrics.

For `diabetes_dataset`, the narrower-bandwidth candidate was selected and slightly improved validation accuracy and test AUC-ROC. Held-out test accuracy was unchanged at `0.9700`, while XGBoost remained higher at `0.9727` in the first-round baseline results. This suggests that the second-round xRFM grid did not materially change the classification conclusion for this dataset. The CPU cost also remains a practical concern: xRFM training took about 199.7 seconds in round two, compared with sub-second training times for the saved Random Forest and XGBoost runs.

For `student_dropout_academic_success`, round-two tuning improved held-out test accuracy from `0.7571` to `0.7695`, slightly exceeding the saved XGBoost test accuracy of `0.7684`. However, validation accuracy was unchanged, and test AUC-ROC decreased from `0.8718` to `0.8692`, below the saved Random Forest and XGBoost AUC-ROC values. This should be reported as a mixed result, not as a decisive improvement.

For `winequality_white` and `heart_disease_uci`, round-two tuning did not improve the primary validation or held-out test metrics. These datasets were already relatively small, and the changes in performance are small enough that they should not be overinterpreted from a single split.

## Conclusions and Limitations

The second tuning round supports a careful conclusion: targeted xRFM tuning can improve selected metrics on some datasets, but the benefits are modest and not universal. The most useful report narrative is that xRFM was competitive on several datasets and particularly promising on `online_news_popularity`, while remaining expensive on `diabetes_dataset` and sensitive to dataset-specific tuning.

Several limitations should be kept explicit:

- The comparison uses one seed and one held-out split.
- Candidate selection used validation metrics, while test metrics were used only for final evaluation.
- The round-two grid was intentionally compact for local CPU practicality, so it does not exhaust the full xRFM hyperparameter space.
- Some round-two changes improved one metric while worsening another, especially on `online_news_popularity` and `student_dropout_academic_success`.

These results are therefore best interpreted as project-specific evidence for dataset-dependent xRFM behavior, rather than as a general benchmark claim.

## Result Sources

- Round-one summary: `results/summary/results.csv`
- Round-two xRFM summary: `results/summary/xrfm_round2_results.csv`
- Round-one raw results: `results/raw/*.json`
- Round-two xRFM raw results: `results/raw_round2_xrfm/*.json`
- Round-two grid: `configs/xrfm_grid.yaml`
- Smoke grid reference: `configs/xrfm_smoke_grid.yaml`
- Baseline grid reference: `configs/baseline_grid.yaml`
