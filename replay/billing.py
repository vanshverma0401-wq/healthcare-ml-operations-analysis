"""A bounded, group-disjoint high-cost billing baseline, separate from V4/V5."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['age', 'gender', 'blood_type', 'medical_condition',
            'admission_type', 'insurance_provider']
SEED = 42


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns=lambda c: c.strip().lower().replace(' ', '_')).copy()
    if frame.columns.duplicated().any():
        raise ValueError('duplicate normalized columns')
    required = {'name', 'billing_amount'}
    if not required <= set(frame):
        raise ValueError(f'missing columns: {sorted(required - set(frame))}')
    frame['billing_amount'] = pd.to_numeric(frame['billing_amount'], errors='raise')
    if not np.isfinite(frame['billing_amount']).all():
        raise ValueError('billing_amount must be finite')
    if frame['name'].isna().any() or frame['name'].astype(str).str.strip().eq('').any():
        raise ValueError('name is required only for the private split-group proxy')
    return frame


def name_groups(frame: pd.DataFrame) -> pd.Series:
    # Kept in memory only. This is not a verified patient identifier.
    return frame['name'].astype(str).str.strip().str.casefold()


def bounded_groups(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows < 100:
        raise ValueError('max_rows must be at least 100')
    groups = name_groups(frame)
    counts = groups.value_counts()
    order = sorted(counts.index, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    selected, rows = [], 0
    for key in order:
        count = int(counts[key])
        if rows + count <= max_rows:
            selected.append(key); rows += count
    if len(selected) < 10:
        raise ValueError('not enough complete name-proxy groups for a bounded split')
    return frame.loc[groups.isin(selected)].reset_index(drop=True)


def split_groups(frame: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=1, train_size=.70, random_state=SEED)
    train_pos, rest_pos = next(splitter.split(frame, groups=name_groups(frame)))
    train, rest = frame.iloc[train_pos].copy(), frame.iloc[rest_pos].copy()
    valid_pos, test_pos = next(GroupShuffleSplit(n_splits=1, train_size=.5,
                              random_state=SEED + 1).split(rest, groups=name_groups(rest)))
    valid, test = rest.iloc[valid_pos].copy(), rest.iloc[test_pos].copy()
    group_sets = [set(name_groups(x)) for x in (train, valid, test)]
    if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError('split-group overlap')
    return train, valid, test


def fit_cutoff(training: pd.DataFrame) -> float:
    return float(training['billing_amount'].quantile(.85))


def build_model() -> Pipeline:
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preparation = ColumnTransformer([('numeric', numeric, ['age']),
                    ('categorical', categorical, [x for x in FEATURES if x != 'age'])])
    return Pipeline([('prepare', preparation),
                     ('model', LogisticRegression(class_weight='balanced', max_iter=1000,
                                                   random_state=SEED))])


def metrics(actual, probabilities) -> dict:
    if len(np.unique(actual)) != 2:
        raise ValueError('both classes required in each evaluation partition')
    predicted = probabilities >= .5
    return {'positive_rows': int(np.sum(actual)), 'rows': int(len(actual)),
            'threshold': .5,
            'precision': float(precision_score(actual, predicted, zero_division=0)),
            'recall': float(recall_score(actual, predicted, zero_division=0)),
            'positive_f1': float(f1_score(actual, predicted, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(actual, predicted)),
            'roc_auc': float(roc_auc_score(actual, probabilities)),
            'average_precision': float(average_precision_score(actual, probabilities)),
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(actual, predicted, labels=[0, 1]).ravel().tolist()}


def evaluate(frame: pd.DataFrame, max_rows: int = 10000) -> dict:
    normalized = normalize_frame(frame)
    missing = set(FEATURES) - set(normalized)
    if missing:
        raise ValueError(f'missing features: {sorted(missing)}')
    unique = normalized.drop_duplicates().reset_index(drop=True)
    sample = bounded_groups(unique, max_rows)
    train, valid, test = split_groups(sample)
    cutoff = fit_cutoff(train)
    labels = [x['billing_amount'].ge(cutoff).astype(int).to_numpy() for x in (train, valid, test)]
    models = {'prior_baseline': DummyClassifier(strategy='prior'), 'fixed_logistic': build_model()}
    results = {}
    for name, model in models.items():
        model.fit(train[FEATURES], labels[0])
        results[name] = {key: metrics(y, model.predict_proba(part[FEATURES])[:, 1])
                        for key, part, y in [('validation', valid, labels[1]), ('test', test, labels[2])]}
    return {'analysis_date_local': '2026-11-10',
            'scope': 'Post-submission bounded retrospective synthetic-data benchmark; not an original V4/V5 replay or clinical validation',
            'input_rows': len(normalized), 'duplicate_rows_removed': len(normalized)-len(unique),
            'sample_rows': len(sample), 'max_rows': max_rows,
            'sample_policy': 'Whole normalized-name groups ordered by SHA-256, independent of outcome',
            'split_policy': '70/15/15 approximate row fractions, disjoint normalized-name proxy groups; seeds 42/43',
            'partition_rows': {k: len(v) for k, v in [('train', train), ('validation', valid), ('test', test)]},
            'group_overlap': 0, 'features': FEATURES,
            'target': 'billing_amount >= training-only 85th percentile', 'training_cutoff': cutoff,
            'model_selection': 'Prespecified baseline and logistic settings; no validation/test tuning; fixed probability threshold 0.5',
            'models': results,
            'limits': ['Normalized names are only a conservative split proxy, not stable verified patient IDs.',
                       'Synthetic retrospective data; feature availability at admission is assumed, not verified by field timestamps.',
                       'Historical records were previously studied; this is not a new external blinded test.',
                       'No claim of treatment utility, clinical validity, HIPAA compliance or deployed performance.'],
            'environment': {'python': platform.python_version(), **{n: importlib.metadata.version(n)
                            for n in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'cloudpickle', 'narwhals']}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Local permitted CSV; never uploaded')
    parser.add_argument('--output', type=Path, required=True, help='Aggregate-only JSON report')
    parser.add_argument('--max-rows', type=int, default=10000)
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error('Output must differ from the source CSV, including symlink aliases')
    source = args.input.read_bytes()
    result = evaluate(pd.read_csv(args.input), args.max_rows)
    result['input_sha256'] = hashlib.sha256(source).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'sample_rows': result['sample_rows'],
                      'partition_rows': result['partition_rows'], 'group_overlap': 0}))


if __name__ == '__main__':
    main()
