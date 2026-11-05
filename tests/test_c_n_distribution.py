"""Tests for saved N-grid reuse. Written only; not executed for this task."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.c_n_distribution import attach, select_source
from module.pipeline import resolve_targets


class SavedNDistributionTests(unittest.TestCase):
    def artifact(self):
        return dict(grid=np.array([[1., 2.], [3., 4.]]),
                    spatial=dict(grid_spacing_m=10., property_mask_bounds=dict(xmin=0., ymin=0., xmax=10., ymax=10.)),
                    metadata=dict(source_depth_m=5.))

    def test_same_xy_uses_fixed_depth_map(self):
        frame = pd.DataFrame(dict(x=[1., 1.], y=[1., 1.], depth=[2., 9.]))
        result = attach(frame, self.artifact())
        self.assertEqual(result.n_value.tolist(), [1., 1.])
        self.assertEqual(result.n_distribution_depth_m.tolist(), [5., 5.])
        self.assertEqual(result.depth.tolist(), [2., 9.])

    def test_missing_or_outside_is_not_filled(self):
        with self.assertRaises(ValueError):
            attach(pd.DataFrame(dict(x=[-1.], y=[0.])), self.artifact())
        artifact = self.artifact(); artifact['grid'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            attach(pd.DataFrame(dict(x=[0.], y=[0.])), artifact)

    def test_phi_precedes_c_even_when_requested_last(self):
        self.assertEqual(resolve_targets({}, ['c', 'phi', 'gamma']), ('phi', 'c', 'gamma'))
        self.assertEqual(resolve_targets({}, ['c']), ('c',))

    def test_current_failed_phi_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'run_summary.json').write_text(json.dumps(dict(status='failed')))
            with patch('module.c_n_distribution.RESULTS', run/'unused'):
                with self.assertRaisesRegex(ValueError, 'not complete'):
                    select_source({'run_directory': 'old'}, current_run=run)

    def test_explicit_source_without_chunks_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'run_summary.json').write_text('{"status":"complete"}')
            (run/'config.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):
                select_source({'run_directory': str(run)})

