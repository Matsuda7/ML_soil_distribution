"""Saved-stage and model-dispatch tests. Definitions only; not executed here."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from module.run_artifacts import trained_run, archive_grid
from module.model_backend import selected_model


class RunArtifactsTests(unittest.TestCase):
    def test_unimplemented_model_never_falls_back(self):
        with patch('module.model_backend.load_model_settings',return_value={'model':'tabpfn'}):
            with self.assertRaisesRegex(ValueError,'not implemented'):
                selected_model()

    def test_explicit_run_must_match_target_and_estimator(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            (run/'training_status.json').write_text(json.dumps({'status':'trained','target':'phi','estimator':'other'}))
            with self.assertRaises(ValueError):trained_run('phi','xgboost',run)

    def test_old_result_without_training_marker_is_not_assumed_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'stage-03'):
                trained_run('phi','xgboost',tmp)

    def test_grid_archive_preserves_previous_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'grid').mkdir();(run/'grid/value.txt').write_text('old')
            (run/'config.json').write_text('{}');archive_grid(run)
            self.assertFalse((run/'grid').exists())
            archived=list((run/'prediction_archive').iterdir())
            self.assertEqual((archived[0]/'grid/value.txt').read_text(),'old')
            self.assertTrue((archived[0]/'config.json').is_file())
