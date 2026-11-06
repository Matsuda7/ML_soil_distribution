"""Tests for saved N-grid reuse. Written only; not executed for this task."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.c_n_distribution import attach, select_source, inspect_grid_batch, raise_grid_diagnostics
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


    def test_diagnostics_separate_coordinate_and_n_failures(self):
        spatial=self.artifact()['spatial'];grid=np.full((2,2),np.nan)
        frame=pd.DataFrame(dict(x=[0.,20.,10.,float('nan'),0.],y=[0.,0.,0.,0.,10.],
                                predicted_n_value=[51.,4.,float('inf'),5.,-1.]))
        _,_,_,report=inspect_grid_batch(frame,spatial,(2,2),grid,row_offset=100000)
        self.assertEqual(report['counts']['outside_grid_index_range'],1)
        self.assertEqual(report['counts']['non_finite_or_non_numeric_coordinates'],1)
        self.assertEqual(report['counts']['non_finite_or_non_numeric_n'],1)
        self.assertEqual(report['counts']['n_above_50'],1)
        self.assertEqual(report['counts']['n_below_zero'],1)
        self.assertEqual(report['examples']['n_above_50'][0]['csv_row_number'],100002)

    def test_diagnostics_capture_alignment_and_duplicate_rows(self):
        spatial=self.artifact()['spatial'];grid=np.full((2,2),np.nan);grid[1,1]=7.
        frame=pd.DataFrame(dict(x=[0.,0.,10.,1.],y=[0.,0.,10.,0.],predicted_n_value=[1.,2.,3.,4.]))
        _,_,_,report=inspect_grid_batch(frame,spatial,(2,2),grid)
        self.assertEqual(report['counts']['duplicate_coordinates_in_batch'],3)
        self.assertEqual(report['counts']['duplicate_coordinates_in_previous_batches'],1)
        self.assertEqual(report['counts']['coordinates_not_aligned_to_grid'],1)

    def test_failure_diagnostics_are_strict_json_and_csv(self):
        spatial=self.artifact()['spatial'];grid=np.full((2,2),np.nan)
        frame=pd.DataFrame(dict(x=[0.],y=[0.],predicted_n_value=[float('inf')]))
        _,_,_,report=inspect_grid_batch(frame,spatial,(2,2),grid)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'validation_error.json'):
                raise_grid_diagnostics('input.csv.gz',Path(tmp),report)
            payload=json.loads((Path(tmp)/'validation_error.json').read_text())
            self.assertEqual(payload['counts']['non_finite_or_non_numeric_n'],1)
            self.assertTrue((Path(tmp)/'validation_error_rows.csv').is_file())

    def test_cropping_ignores_only_known_outside_n_cells(self):
        spatial=self.artifact()['spatial'];grid=np.full((2,2),np.nan)
        frame=pd.DataFrame(dict(x=[0.,0.,10.],y=[-10.,0.,0.],predicted_n_value=[999.,2.,3.]))
        xi,yi,values,report=inspect_grid_batch(frame,spatial,(2,2),grid,crop_outside=True)
        self.assertEqual(values.tolist(),[2.,3.])
        self.assertEqual(xi.tolist(),[0,1])
        self.assertEqual(yi.tolist(),[0,0])
        self.assertEqual(report['ignored_counts']['outside_requested_grid'],1)
        self.assertFalse(any(report['counts'].values()))

    def test_cropping_preserves_errors_and_original_csv_row_numbers(self):
        spatial=self.artifact()['spatial'];grid=np.full((2,2),np.nan)
        frame=pd.DataFrame(dict(x=[0.,0.,float('nan')],y=[-10.,0.,0.],predicted_n_value=[4.,51.,2.]))
        _,_,_,report=inspect_grid_batch(frame,spatial,(2,2),grid,row_offset=100,crop_outside=True)
        self.assertEqual(report['counts']['n_above_50'],1)
        self.assertEqual(report['counts']['non_finite_or_non_numeric_coordinates'],1)
        self.assertEqual(report['examples']['n_above_50'][0]['csv_row_number'],103)
