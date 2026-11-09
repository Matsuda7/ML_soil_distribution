"""Small fake-model tests. No TabPFN/torch import, weights, or research data."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tabpfn_benchmark', HERE/'benchmark_tabpfn.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def config(self):
        return bench.read_json(HERE/'config.json')

    def test_invalid_configuration(self):
        for value in (0, -1, 1.5, '8', True):
            config = self.config(); config['cpu_threads'] = value
            with self.assertRaises(ValueError): bench.validate(config)
        for value in (None, 1, 128):
            config = self.config(); config['cpu_threads'] = value
            bench.validate(config)

    def test_balanced_reproducible_order(self):
        config = self.config()
        first = bench.schedule(config)
        self.assertEqual(first, bench.schedule(config))
        self.assertEqual(len(first), 24)
        for repeat in range(1, 4):
            self.assertEqual(len({(x['fit_mode'], x['batch_size']) for x in first if x['repeat'] == repeat}), 8)

    def test_cache_rebuild_changes_only_mode_and_uses_all_rows(self):
        import numpy as np
        class Fake:
            def __init__(self, **params): self.params = params
            def fit(self, x, y): self.x, self.y = x.copy(), y.copy()
        params = dict(fit_mode='low_memory', random_state=42, n_estimators=8,
                      memory_saving_mode=True, categorical_features_indices=[2], model_path='local.ckpt')
        x, y = np.arange(30).reshape(10, 3), np.arange(10)
        cached = bench.fit_cached_model(Fake, params, x, y)
        self.assertEqual(params['fit_mode'], 'low_memory')
        self.assertEqual(cached.params, dict(params, fit_mode='fit_with_cache'))
        np.testing.assert_array_equal(cached.x, x)
        np.testing.assert_array_equal(cached.y, y)

    def test_failure_csv_retains_required_columns(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            bench.save_csv(path, [dict(status='failed', error='fake failure')])
            with (path/'results.csv').open() as stream:
                columns = csv.DictReader(stream).fieldnames
            for name in ('fit_mode','batch_size','cpu_threads','prediction_rows',
                         'model_load_seconds','cache_build_seconds','inference_seconds',
                         'points_per_second','peak_memory_mb','max_abs_prediction_diff','status'):
                self.assertIn(name, columns)

    def test_time_estimate(self):
        row = bench.extrapolate(10000, 500, 3, 7, 20000000)
        self.assertEqual(row['points_per_second'], 20)
        self.assertEqual(row['estimated_total_seconds_20m'], 1000010)

    def test_difference_and_raw_clipping(self):
        import numpy as np
        row = bench.differences(np.array([-2., 4.]), np.array([-1., 4.]), 1e-5, 1e-5)
        self.assertEqual(row['max_abs_prediction_diff'], 1)
        self.assertEqual(row['max_abs_corrected_diff'], 0)
        self.assertFalse(row['predictions_close'])

    def test_reservoir_is_bounded_and_reproducible(self):
        import numpy as np
        def sample():
            rng = np.random.default_rng(42)
            points, keys = np.empty((0, 2)), np.empty(0)
            all_points = np.arange(200).reshape(100, 2)
            for chunk in np.array_split(all_points, 4):
                points, keys = bench.reservoir_update(points, keys, chunk, rng.random(len(chunk)), 10)
                self.assertLessEqual(len(points), 10)
            return points
        self.assertTrue(np.array_equal(sample(), sample()))
        self.assertEqual(len(np.unique(sample(), axis=0)), 10)

    def test_production_batch_helper_with_fake_model(self):
        import numpy as np
        from module.tabpfn_regression import predict_batches
        class FakeModel:
            def predict(self, x): return x[:, 0] * 2
        x = np.arange(30).reshape(10, 3)
        for batch in (1, 3, 256):
            np.testing.assert_array_equal(predict_batches(FakeModel(), x, batch), x[:, 0]*2)
        self.assertNotIn('tabpfn', sys.modules)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
