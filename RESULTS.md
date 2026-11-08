# Recorded results

These are the team's saved academic results, recovered from the final project
materials in November 2026. They were not regenerated for this publication.
The project contains **billing experiments and an operations/capacity analysis**;
their targets, observation units and evaluation procedures are different.

## Billing: original V4 final analysis

The final billing notebook cleaned 55,500 synthetic records to **54,966 rows**,
removing 534 duplicates. It reused an 80/20 index split across its three tasks:
43,972 training rows and 10,994 test rows. These are historical comparison scores,
rather than an independently validated clinical or financial prediction system.

| Billing-category model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Multinomial Logistic Regression | 0.3377 | 0.3372 | 0.3373 |
| Random Forest | 0.4287 | **0.4286** | 0.4286 |
| Extra Trees | 0.4280 | 0.4280 | 0.4279 |

Random Forest was the presentation's selected V4 billing-category model. The
separate V3 experiment reached macro F1 **0.4340**; it is an earlier result, not
the score of the selected V4 model. Earlier raw-billing regression had
R-squared approximately zero, motivating the move to Low/Medium/High categories.

| Insurance-provider model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Multinomial Logistic Regression | 0.1953 | 0.1932 | 0.1933 |
| Random Forest | 0.3140 | 0.3140 | 0.3140 |
| Extra Trees | **0.3145** | **0.3145** | **0.3145** |

Insurance-provider prediction remained a secondary exploratory task. The labels
describe synthetic records; the result does not establish an insurer-selection
or patient-routing capability.

Read the [V4 source and saved outputs](FULL_BILLING_REPORT.md#v4-cell-0), its
[notebook](notebooks/billing_v4_final.ipynb), and the
[complete figure gallery](FIGURE_GALLERY.md).

## Billing: V5 post-presentation extension

V5 introduced medication/condition interaction and a test-result intensity
feature, stratified splitting, cross-validation, randomized hyperparameter
search and XGBoost. It recorded several **distinct evaluation populations**.

| V5 experiment | Evaluation population | Saved macro F1 |
| --- | --- | ---: |
| Initial Random Forest | Five-fold stratified CV within the training partition | 0.4104 |
| Random Forest randomized search | Best five-fold CV score within the training partition | 0.4118 |
| XGBoost | Five-fold stratified CV within the training partition | 0.3544 |

The search score is the best score found during parameter selection, not a new
held-out test result. The notebook also compared feature sets using five-fold CV
over the **full working dataframe**, including rows belonging to its earlier
holdout. That separate comparison recorded:

| Feature set | Model | Mean macro F1 | Fold-score standard deviation |
| --- | --- | ---: | ---: |
| Baseline | Extra Trees | 0.430323 | 0.001310 |
| Enhanced | Random Forest | 0.430249 | 0.002330 |
| Enhanced | Extra Trees | 0.430066 | 0.002481 |
| Baseline | Random Forest | 0.428791 | 0.003767 |
| Baseline | Logistic Regression | 0.335450 | 0.003400 |
| Enhanced | Logistic Regression | 0.330966 | 0.004164 |

These rows should not be ranked directly against the train-partition CV scores
or the V4 test scores as if all used the same evaluation population. The added
complexity did not establish a strong or reliable predictive signal.
See the [V5 source and recorded outputs](FULL_BILLING_REPORT.md#v5-cell-0) and
[V5 notebook](notebooks/billing_v5_extension.ipynb).

## High-cost experiments: a source inconsistency retained

The V4 notebook reports balanced Logistic Regression class-1 F1 **0.2367**.
The later billing presentation contains a revised high-cost comparison:

| Model in the later presentation table | Accuracy | Positive-class precision | Positive-class recall | Positive-class F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced Logistic Regression | 0.4984 | 0.1491 | 0.4982 | 0.2296 | 0.4994 |
| Balanced Random Forest | 0.8641 | 0.9755 | 0.0966 | 0.1758 | 0.6128 |
| Weighted two-model ensemble | 0.6157 | 0.1989 | 0.5158 | 0.2871 | 0.6106 |
| XGBoost | 0.5761 | 0.1612 | 0.4344 | 0.2352 | 0.5056 |
| GBM | 0.5443 | 0.1515 | 0.4429 | 0.2258 | 0.6106 |

The later deck's summary slide still repeats the older **0.2367** result, while
the detailed table gives **0.2296**. Its prose refers to a three-model ensemble,
but the table names a **two-model** ensemble. These differences have not been
silently reconciled into a single “final best” number.

The [high-cost source](src/billing/predict_high_cost_cases.py) searches thresholds
and ensemble weights against `y_test`. Its tuned comparisons therefore reuse
test labels for selection. High accuracy for the Random Forest also coexists
with very low positive-class recall. These experiments remain a limitation
analysis, not evidence of a deployable high-cost screening model.

## Operations and capacity

The operations stream retains 55,500 synthetic encounter rows and constructs
**7,418 artificial-coalition daily rows** across four groups, dated May 8, 2019
through June 5, 2024. These counts and observation units differ from the cleaned
record-level billing analysis. No measured staffing or real bed-capacity data were
provided.

The [refined source](src/operations/refined.py) uses chronological partitions
within each coalition. Its final saved locked-policy summary records:

| Intended horizon | Test rows | Recall | Precision | Flagged share | Combined review-score test ROC-AUC | AUTO |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 day | 1,484 | 0.9655 | 0.1207 | 0.1563 | 0.9738 | Disabled |
| 2 days | 1,484 | 0.9310 | 0.0982 | 0.1853 | 0.9366 | Disabled |
| 3 days | 1,483 | 0.8276 | 0.0866 | 0.1868 | 0.9228 | Disabled |
| 4 days | 1,482 | 0.6897 | 0.0722 | 0.1869 | 0.8555 | Disabled |
| 5 days | 1,481 | 0.6207 | 0.0700 | 0.1735 | 0.8500 | Disabled |
| 6 days | 1,480 | 0.5862 | 0.0627 | 0.1831 | 0.8125 | Disabled |
| 7 days | 1,480 | 0.6552 | 0.0722 | 0.1777 | 0.7709 | Disabled |

Each horizon's saved test set contains only **29 positive rows**. At the one-day
setting, 28 positives were flagged and one was missed, alongside **204 false
positives and 1,251 true negatives**. High recall therefore coexists with about
12.1% precision. The combined review score is the maximum of the gradient-boosting
and SVM positive-class scores; its AUC column refers to those saved test scores,
not to a separate validation partition. All seven AUTO settings remain disabled.

The capacity definition uses the complete date range, some rolling windows can
cross coalition boundaries, and the retained script lacks an interactive
threshold-lock definition needed for a clean replay. These recorded scores do
not establish a validated operational forecast. The source's intended day
horizons also require an explicit calendar reindex before exact calendar-day
alignment can be assumed for every row.

The [operations source audit](technical/operations_source_audit.json) preserves
all confusion counts, thresholds and model-specific AUC values. The
[operations gallery](OPERATIONS_GALLERY.md) includes all ten unique saved figures,
including baseline diagnostics; the baseline actually uses a random stratified
split despite its time-safe source comment. Read the
[methods review](METHODS_AND_LIMITATIONS.md#operations-and-capacity-workflow)
alongside either set of charts.
