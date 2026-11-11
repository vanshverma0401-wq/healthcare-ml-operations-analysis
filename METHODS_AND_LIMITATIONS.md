# Methods and limitations

This collaborative coursework asks which financial and operational outcomes
can be learned from a synthetic healthcare dataset. Its value is the documented
workflow: cleaning, feature construction, model comparison, evaluation and
recognition of weak or unstable results. The billing and capacity streams are
separate analyses rather than components of a deployed clinical system.

## Source data and preparation

The upstream [Healthcare Dataset by Prasad Patil](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
is described by its publisher as synthetic, generated with Faker, and licensed
**CC0: Public Domain**. The source API was checked on November 8, 2026:
version 2, last updated May 8, 2024. See [data attribution and usage](DATA_AND_USAGE.md)
for the publication's handling of the source.

The team standardized text and column names, converted dates and numeric fields,
removed duplicate rows and engineered length of stay, admission timing, age bands
and budget labels. The V4 billing notebook retained **54,966 of 55,500 rows**.
The four hospital-system labels are an added grouping field in the project data;
they do not establish observations from four actual healthcare systems.

Raw records, name-like previews and private institutional links are excluded
from this repository. The public material preserves the methods and aggregate
results while recording the specific sanitization of notebook content.

## V4 billing workflow

The [V4 final notebook](notebooks/billing_v4_final.ipynb) defines three tasks:
billing tertile classification, insurance-provider classification and binary
high-cost classification. It compares multinomial Logistic Regression, Random
Forest and Extra Trees for the multiclass tasks, with class-weighted alternatives
for the binary task.

A shared unstratified 80/20 split uses fixed row indices and random seed 42.
Preprocessing combines median numerical imputation and scaling with
most-frequent categorical imputation and one-hot encoding. High-cardinality
doctor/hospital frequency maps are learned **from training rows only** and then
applied to test rows. This is separate from an earlier full-dataset encoding
approach mentioned in historical materials.

Billing-category tertiles and the top-15-percent high-cost cutoff are constructed
from the full working dataframe before splitting. A future evaluation should
learn or prespecify those boundaries using training data alone. Repeated model
and version comparisons also do not create a fresh untouched test set.

Some predictors, including length of stay and recorded test results, may only be
available after admission or near discharge. Their use does not establish a
prediction that could have been made at admission. A practical forecasting
claim would need an explicit decision time and a feature-availability audit.

## V5 validation and feature experiments

The [V5 extension](notebooks/billing_v5_extension.ipynb) adds a medication/condition
interaction, test-result intensity, five-fold stratified CV, randomized Random
Forest search and an XGBoost comparison. Its initial CV and tuning sections use
the training partition. A different feature-comparison loop uses the full
dataframe. [Recorded results](RESULTS.md) labels those populations separately.

The feature loop silently selects only requested columns that are available.
Its “baseline” label therefore describes that notebook's available subset and
must not be assumed identical to the full V4 feature set. The best randomized
search score is a selection score; an independent locked evaluation was not
performed for this publication.

## High-cost selection and metric interpretation

The [high-cost script](src/billing/predict_high_cost_cases.py) includes
training-only numerical outlier filtering, class balancing, multiple tree/linear
models, probability threshold searches and weighted ensembles. However, threshold
and ensemble-weight searches repeatedly optimize F1 using `y_test`. Reported
tuned scores from that workflow are not independent test estimates.

Macro F1 averages class-specific F1 values equally. Weighted F1 weights them by
support. Binary class-1 F1 concerns the high-cost class only; these metrics are
not interchangeable. Accuracy can remain high while most high-cost examples are
missed. The source's newer and older high-cost figures conflict, as documented
in [Recorded results](RESULTS.md#high-cost-experiments-a-source-inconsistency-retained).

## Operations and capacity workflow

The operations branch retains the 55,500 synthetic records without the billing
branch's deduplication step. It aggregates admission/discharge and occupied-room
information into 7,418 daily rows across four artificial coalitions. The baseline
room key combines system and room number without hospital qualification; the
refined source instead constructs `room_uid` from hospital ID and room number.
Both remain synthetic occupancy proxies rather than a verified hospital bed census.

The [baseline](src/operations/baseline.py) compares Logistic Regression and an
RBF SVM for low/normal/high future-utilization labels. Its actual split is random
and stratified, despite a time-safe comment. Its low/high cutoffs use full-series
quartiles. Some longer-horizon baseline charts plot manually entered historical
metric arrays instead of recomputing the models in that script.

The [refined analysis](src/operations/refined.py) compares histogram gradient
boosting and an RBF SVM, with lag/history features and a manual-review score
defined as the maximum of their positive-class probabilities. It partitions each
coalition chronologically into earliest 70%, next 10% and latest 20% rows.
Capacity is nevertheless a constructed quantity:
`ceil(full-series 95th-percentile occupied rooms / 0.85)`, with a minimum of 20.
The high-utilization target uses a ratio **greater than or equal to 0.85**.
Neither number is observed staffed capacity.

Several issues limit the recorded forecast interpretation:

- Capacity calibration and artificial-coalition assignment use the complete time
  range before partitioning, so held-out information influences the task.
- Some histories apply a plain-Series rolling window after a grouped shift,
  allowing carry-over across coalition boundaries.
- There is no horizon-sized purge at partition boundaries to exclude rows whose
  shifted targets cross into the next partition.
- Daily occupancy rows are not explicitly reindexed to every coalition/date
  combination. A row shift does not by itself prove an exact calendar-day lead.
- Positive examples are sparse: the one-day validation partition has nine and
  its saved test partition has 29. Final review precision is low, and all seven
  AUTO settings remain disabled.
- `review_lock_by_horizon_df` is referenced but not defined in the supplied
  refined script. Saved results depend on retained interactive state; a clean
  top-to-bottom replay has not been established.

[Recorded results](RESULTS.md#operations-and-capacity) presents the saved review
workload and precision alongside recall and AUC. The
[source audit](technical/operations_source_audit.json) and
[operations gallery](OPERATIONS_GALLERY.md) preserve the original aggregate
evidence without converting its actionability labels into validated claims.

## Reproduction and publication scope

The original notebooks, scripts and charts are historical source and saved
results. No original-model retraining or live-data integration was performed for
the November 8 publication. The [November 10 billing extension](REPLAY_EXTENSION.md)
is a separately implemented, bounded benchmark with its own tested environment;
it does not establish clean-kernel execution of the original models. The source environment
and the project-added hospital-system input must be reconstructed deliberately
before attempting a new run; source inspection does not prove runtime success.

The [billing report](FULL_BILLING_REPORT.md) preserves the substantive notebook
content, and the [figure manifest](technical/figure_manifest.json) records saved
image provenance. Team attribution and the distinction between original work and
new publication utilities are retained.

These synthetic-data experiments do not establish patient-level clinical
validity, HIPAA compliance, a hospital deployment or observed staffing outcomes.
A next research iteration should prespecify feature timing and targets, use
training-only preparation and model selection, lock an independent evaluation,
and report uncertainty and failure costs alongside aggregate scores.
