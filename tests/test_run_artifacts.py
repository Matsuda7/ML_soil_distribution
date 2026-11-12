"""Saved-stage and model-dispatch tests. Definitions only; not executed here."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from module.run_artifacts import trained_run, archive_grid, write_json, read_json
from module.model_backend import selected_model, get_backend


class RunArtifactsTests(unittest.TestCase):
    def test_json_retries_transient_lock_and_keeps_old_json_until_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'progress.json'
            write_json(path, {'status': 'old'})
            replace = Path.replace
            calls = []

            def locked_once(source, destination):
                calls.append(source)
                self.assertEqual(read_json(destination), {'status': 'old'})
                if len(calls) == 1:
                    raise PermissionError('temporary lock')
                return replace(source, destination)

            with patch.object(Path, 'replace', locked_once), patch('module.run_artifacts.sleep') as wait:
                write_json(path, {'status': '推論中'})
            self.assertEqual(read_json(path), {'status': '推論中'})
            wait.assert_called_once_with(0.1)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_json_persistent_lock_is_reported_and_old_json_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'progress.json'
            write_json(path, {'status': 'old'})
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')) as replace, \
                    patch('module.run_artifacts.sleep') as wait:
                with self.assertRaises(PermissionError):
                    write_json(path, {'status': 'new'})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(wait.call_count, 5)
            self.assertEqual(read_json(path), {'status': 'old'})
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_json_other_io_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Path, 'replace', side_effect=OSError('disk error')), \
                    patch('module.run_artifacts.sleep') as wait:
                with self.assertRaises(OSError):
                    write_json(Path(tmp)/'progress.json', {})
            wait.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_tabpfn_stage03_dispatch_is_explicit(self):
        with patch('module.model_backend.load_model_settings',return_value={'model':'tabpfn'}):
            self.assertEqual(selected_model(),'tabpfn')
            with patch('module.tabpfn_training.train_cohesion',return_value={'status':'trained'}) as train:
                self.assertEqual(get_backend().train('c'),{'status':'trained'})
                train.assert_called_once_with(prepared_data=None)

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
