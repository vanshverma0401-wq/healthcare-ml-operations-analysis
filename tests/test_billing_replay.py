"""Contracts for the separately labeled post-submission billing replay."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stderr
import io
import pandas as pd
from replay.billing import FEATURES, fit_cutoff, normalize_frame, split_groups, build_model, bounded_groups


class BillingReplayTests(unittest.TestCase):
    def test_cli_rejects_output_alias_of_source_before_reading(self):
        from replay.billing import main
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'private.csv'
            frame = pd.DataFrame({c: ['Fixture'] * 400 for c in FEATURES if c != 'age'})
            frame['age'] = 40
            frame['name'] = [f'Fixture Person {i}' for i in range(400)]
            frame['billing_amount'] = range(400)
            frame.to_csv(source, index=False)
            before = source.read_bytes()
            alias = Path(folder) / 'report.json'
            alias.symlink_to(source)
            with patch('sys.argv', ['billing', '--input', str(source), '--output', str(alias)]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(source.read_bytes(), before)

    def test_cutoff_uses_only_training_amounts(self):
        train = pd.DataFrame({'billing_amount': [10., 20., 30., 40.]})
        cutoff = fit_cutoff(train)
        self.assertAlmostEqual(cutoff, 35.5)
        self.assertNotEqual(cutoff, pd.Series([10., 20., 30., 40., 100000.]).quantile(.85))

    def test_name_proxy_groups_never_cross_partitions(self):
        df = pd.DataFrame({'name': [f' person {i} ' for i in range(40) for _ in range(2)]})
        train, valid, test = split_groups(df)
        groups = [set(x['name'].str.strip().str.casefold()) for x in [train, valid, test]]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(sum(map(len, [train, valid, test])), len(df))

    def test_decision_features_exclude_outcomes_and_identifiers(self):
        forbidden = {'billing_amount', 'length_of_stay', 'discharge_date', 'test_results',
                     'medication', 'name', 'doctor', 'hospital', 'room_number'}
        self.assertFalse(set(FEATURES) & forbidden)

    def test_bounded_sampling_keeps_whole_groups(self):
        frame = pd.DataFrame({'name': [f'Fixture {i}' for i in range(40) for _ in range(3)]})
        sample = bounded_groups(frame, 100)
        self.assertLessEqual(len(sample), 100)
        self.assertTrue(sample.groupby('name').size().eq(3).all())

    def test_normalized_column_names_and_nonfinite_target_rejected(self):
        frame = pd.DataFrame({'Name': ['Fixture Person'], 'Billing Amount': [float('inf')]})
        with self.assertRaisesRegex(ValueError, 'finite'):
            normalize_frame(frame)

    def test_preprocessor_fits_training_median_and_handles_unseen_category(self):
        frame = pd.DataFrame({name: ['A', 'B', 'A', 'B'] for name in FEATURES if name != 'age'})
        frame['age'] = [10., 20., 30., 40.]
        model = build_model()
        model.fit(frame, [0, 0, 1, 1])
        imputer = model.named_steps['prepare'].named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(float(imputer.statistics_[0]), 25.)
        held = frame.iloc[:1].copy(); held['age'] = 100000.; held['gender'] = 'UNSEEN'
        model.predict_proba(held)
        self.assertEqual(float(imputer.statistics_[0]), 25.)


if __name__ == '__main__':
    unittest.main()
