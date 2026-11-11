# Post-submission billing benchmark

**Added November 10, 2026.** This is a small new evaluation extension built after
reviewing the authentic final-project archive. It leaves the original team
notebooks, model scripts, saved charts and reports unchanged. It is not a
clean-kernel rerun of V4/V5 or the occupancy branch.

The extension demonstrates **train-only preprocessing, group-aware splitting,
leakage checks, baseline comparison, imbalanced evaluation, reproducible Python
CLIs and automated tests**. Its result is useful precisely because it exposes
weak predictive signal instead of presenting a better-looking historical score.

## Source and row disposition

The original shared archive was rechecked November 10, 2026; its SHA-256 remains
`d0feb94224f1fbd709483971ee0a2e7f4ebf72c0d82dd871a68047effc8efcb1`.
The retained cleaned CSV used for this benchmark has SHA-256
`a91168b628cab7ae07156fa8e1c86bc0dd4e23877ee51ba16eb41f608176ad25`.
Private sharing links and record-level data are excluded.

| Stage | Rows | Interpretation |
| --- | ---: | --- |
| Original synthetic source | 55,500 | Upstream population used by the original project |
| Original billing cleanup | 54,966 | V4 removed 534 exact duplicates |
| Retained cleaned CSV used here | 54,966 | Already cleaned; this run found **0** additional duplicate rows |
| Bounded new benchmark | 10,000 | Complete normalized-name proxy groups, chosen by deterministic SHA-256 ordering |
| Training / validation / test | 6,972 / 1,507 / 1,521 | No normalized-name proxy group crosses a partition |

Names are used **only in memory as a conservative split-group proxy** and are
never predictors or exported. Names are not stable verified patient IDs; name
collisions can group different synthetic people, while spelling differences
could leave related records separate. Exact-duplicate removal and proxy-group
separation do not establish patient-level independence.

## Evaluation design

1. Normalize schema, require finite billing amounts, remove exact duplicate rows.
2. Select at most 10,000 rows while retaining complete name-proxy groups.
3. Partition groups approximately 70/15/15 using seeds 42/43.
4. Define high cost as billing **at or above the training-only 85th percentile**,
   here 43,116.54 in the source's amount units. Apply that same cutoff to both
   held-out partitions.
5. Use only age, gender, blood type, medical condition, admission type and
   insurance provider. Exclude billing, length of stay, discharge, medication,
   test outcomes and high-cardinality person/provider/room identifiers.
6. Fit median imputation, scaling and categorical encoding on training rows only.
   Compare a class-prior baseline and one fixed class-balanced Logistic Regression.
7. Report validation and test metrics at a fixed probability threshold of 0.5.
   No validation/test search or threshold tuning was performed. Validation is
   retained for inspection rather than used to select a winner.

This is a retrospective **administrative-feature** benchmark. The dataset has no
field-level availability timestamps; even medical condition or insurance may
not be final at admission. The feature exclusions therefore reduce obvious
post-outcome leakage without establishing a prospective admission-time system.

## Actual bounded result

[Machine-readable aggregate report](technical/replay/billing_benchmark_2026-11-10.json)
contains exact counts, settings, source hash and environment.

| Test metric (1,521 rows; 199 high-cost) | Prior baseline | Fixed Logistic Regression |
| --- | ---: | ---: |
| Positive-class precision | 0.000 | 0.134 |
| Positive-class recall | 0.000 | 0.503 |
| Positive-class F1 | 0.000 | 0.211 |
| Balanced accuracy | 0.500 | 0.506 |
| ROC-AUC | 0.500 | 0.485 |
| Average precision | 0.131 | 0.134 |

The logistic model's true negatives / false positives / false negatives / true
positives are **674 / 648 / 99 / 100**. Its roughly 50% recall comes with 648
false positives; ROC-AUC is below 0.5. This benchmark does **not** show useful
high-cost discrimination. Scores are not directly comparable to the original
high-cost experiments because predictors, cutoff fitting, population and splits
changed. Historical source records were already studied, so this is not a new
blinded external test.

## How the extension relates to the submission

| Original source issue | Separate extension contract | Still outside this extension |
| --- | --- | --- |
| Full-data high-cost boundary | Cutoff learned from training rows only | Tertile/insurance tasks |
| Repeated test-based threshold/model selection | Prespecified prior/logistic settings and fixed threshold | Nested tuning or a newly collected test set |
| Possible person overlap across row splits | Disjoint normalized-name proxy groups | Verified stable patient identity |
| Post-admission features | Restricted explicit predictor allowlist | Timestamp-verified prospective feature availability |
| Interactive-state dependencies in operations | Standalone billing module and tested CLI | Operations reconstruction and capacity/horizon fixes |

## Reproduce with permitted local inputs

Use an isolated environment; raw data is intentionally absent. See
[data and usage](DATA_AND_USAGE.md) before obtaining data. An arbitrary provider
CSV must match the required schema; the exact cleaned snapshot is identified by
hash above. Run from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
python -m unittest discover -s tests -v
python -m replay.billing --help
python -m replay.billing --input /path/to/healthcare_dataset_cleaned_ready.csv --output technical/replay/local_billing_report.json --max-rows 10000
```

The CLI reads local data, emits aggregate JSON and does not upload anything. Output paths that alias the source CSV, including
symlinks, are rejected before reading or writing the input.
Do not commit raw inputs, split-group names, previews or fitted model artifacts.
The pinned environment was tested on Python 3.12.14. CI exercises synthetic
contract fixtures and existing export tests without private data; it does not
rerun the real-input benchmark or validate clinical use.

Tests check training-only cutoff/preprocessing, complete-group sampling,
disjoint groups, forbidden predictors and invalid target handling. Original
[methods limitations](METHODS_AND_LIMITATIONS.md) and [recorded results](RESULTS.md)
remain applicable to the preserved coursework. There is no clinical validity,
treatment utility, HIPAA, deployment or staffed-capacity claim.

Implementation references: scikit-learn's
[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
and [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
