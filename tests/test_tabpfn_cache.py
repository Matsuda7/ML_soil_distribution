"""Small fake-estimator checks; no torch, real models or research data."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from module.tabpfn_cache import prediction_model
from module.tabpfn_regression import options_from


class CacheTests(unittest.TestCase):
    def test_default_compatibility_and_validation(self):
        self.assertEqual(options_from({})['prediction_fit_mode'], 'low_memory')
        self.assertEqual(options_from({'tabpfn':{'prediction_fit_mode':'fit_with_cache'}})['prediction_fit_mode'], 'fit_with_cache')
        for value in ('bad', None, True, 42):
            with self.assertRaises(ValueError):
                options_from({'tabpfn':{'prediction_fit_mode':value}})

    def test_low_memory_needs_no_training_csv(self):
        class Saved: fit_mode = 'low_memory'
        model = Saved()
        restored, report = prediction_model(None, model, None, None, None, None, 'low_memory')
        self.assertIs(restored, model)
        self.assertEqual(report['cache_build_seconds'], 0.)

    def test_full_rows_same_params_and_immutable_artifacts(self):
        class Saved:
            fit_mode = 'low_memory'
            def __init__(self, **params): self.params = params
            def get_params(self, **kwargs): return self.params.copy()
            def fit(self, x, y): self.x, self.y = x.copy(), y.copy()
        class Transform:
            def transform(self, frame): return frame.to_numpy(dtype=float)
            def fit(self, *args): raise AssertionError('Do not fit saved preprocessing')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = root/'model_dataset.csv'
            table.write_text('row_id,boring_id,depth,target\n0,A,1,10\n1,B,2,20\n2,C,3,30\n')
            digest = hashlib.sha256(table.read_bytes()).hexdigest()
            record = root/'training_data.json'
            record.write_text(json.dumps(dict(directory=str(root),dataset_sha256=digest)))
            metadata = dict(feature_columns=['depth'],numeric_features=['depth'],categorical_features=[],
                            output_sha256={'model_dataset.csv':digest})
            params = dict(fit_mode='low_memory',memory_saving_mode=True,random_state=42,n_estimators=8)
            original = Saved(**params)
            before = {p:p.read_bytes() for p in (table,record)}
            with patch('module.tabpfn_regression.thread_limits',return_value=nullcontext()):
                cached, report = prediction_model(root,original,Transform(),metadata,{'final_fit_rows':3},
                                                 {'tabpfn_version':'2.2.1'},'fit_with_cache')
            self.assertIsNot(cached, original)
            self.assertEqual(cached.params,dict(params,fit_mode='fit_with_cache'))
            self.assertEqual(original.params,params)
            np.testing.assert_array_equal(cached.y,[10,20,30])
            self.assertEqual(len(cached.x),3)
            self.assertEqual(report['cache_training_rows'],3)
            for path, content in before.items(): self.assertEqual(path.read_bytes(),content)
            table.write_text(table.read_text()+'3,D,4,40\n')
            with self.assertRaisesRegex(ValueError,'changed'):
                prediction_model(root,original,Transform(),metadata,{'final_fit_rows':3},
                                 {'tabpfn_version':'2.2.1'},'fit_with_cache')


if __name__ == '__main__': unittest.main()
