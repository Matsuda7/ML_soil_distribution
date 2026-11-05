import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.comparison_data import spatial_folds, load_target
from module.comparison_reporting import metrics
from module.comparison_models import predict_batches, build_model
from module.model_comparison import evaluate_target


class ComparisonTests(unittest.TestCase):
    def config(self):
        config = json.loads(Path('config/xgboost/evaluation.json').read_text())
        config['models'] = ['xgboost']
        config['xgboost'].update(n_estimators=4, n_jobs=1, max_depth=2)
        return config

    def data(self):
        return pd.DataFrame(dict(row_id=np.arange(40), source_row_id=np.arange(40),
            boring_id=np.repeat([f'h{i}' for i in range(20)], 2),
            x=np.repeat(np.arange(20) * 10000 + 100., 2), y=100., depth=5.,
            n_value=np.tile(np.arange(10) + 1., 4)))

    def test_spatial_groups_and_reproducibility(self):
        d = self.data()
        d.loc[2:3, 'boring_id'] = 'h0'  # A borehole spans two blocks.
        d.loc[4:5, 'x'] = d.loc[0, 'x']  # Another hole in the same block.
        a = spatial_folds(d, self.config())
        pd.testing.assert_frame_equal(a, spatial_folds(d, self.config()))
        self.assertEqual(a.groupby('boring_id').fold.nunique().max(), 1)
        self.assertEqual(a.groupby('block').fold.nunique().max(), 1)
        self.assertTrue(a.fold.between(1, 5).all())

    def test_too_few_groups_is_not_random_fallback(self):
        d = self.data(); d.x = 100
        with self.assertRaisesRegex(ValueError, 'independent spatial groups'):
            spatial_folds(d, self.config())

    def test_metrics_and_bias_sign(self):
        score = metrics(np.array([1., 2., 3.]), np.array([2., 3., 4.]))
        self.assertAlmostEqual(score['bias'], 1)
        self.assertAlmostEqual(score['rmse'], 1)
        self.assertAlmostEqual(score['mae'], 1)
        self.assertAlmostEqual(score['r2'], -.5)



    def test_batches_keep_order(self):
        class Model:
            def predict(self, values): return values[:, 0] * 2
        np.testing.assert_array_equal(predict_batches(Model(), np.arange(10).reshape(-1, 1), 3), np.arange(10)*2)



    def test_target_and_learned_features_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nested spatial'):
            load_target('c',dict(data='unused',numeric_features=['n_value'],categorical_features=[]),self.config())




if __name__ == '__main__':
    unittest.main()
