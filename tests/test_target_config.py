import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from module.output_layout import date_directory
from module import pipeline


class TargetConfigTests(unittest.TestCase):
    def test_selection_and_override(self):
        self.assertEqual(pipeline.resolve_targets({}), ('phi', 'c', 'gamma'))
        self.assertEqual(pipeline.resolve_targets({'targets': ['gamma', 'phi', 'gamma']}), ('gamma', 'phi'))
        self.assertEqual(pipeline.resolve_targets({'targets': ['gamma']}, ['phi']), ('phi',))

    def test_invalid_targets(self):
        for value in ([], 'phi', None, ['unknown'], [1], [['phi']]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.resolve_targets({'targets': value})

    def test_plan_and_cli_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.jsonc'
            config.write_text('{"targets": [\n// "phi",\n"gamma",\n]}')
            for extra, expected in (([], ['gamma']), (['--targets', 'phi'], ['phi'])):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), patch.object(pipeline, 'missing_inputs', return_value=[]), patch.object(pipeline, 'run') as run:
                    self.assertEqual(pipeline.main(['--config', str(config), '--plan', *extra]), 0)
                    run.assert_not_called()
                self.assertEqual(json.loads(output.getvalue())['targets'], expected)

    def test_cli_uses_configured_target_for_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.jsonc'
            config.write_text('{"targets": ["phi"]}')
            with patch.object(pipeline, 'run', return_value=0) as run:
                self.assertEqual(pipeline.main(['--config', str(config)]), 0)
                run.assert_called_once_with('all', ('phi',), str(config), False, gamma_kind=None)

    def test_invalid_configuration_stops_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'pipeline.jsonc'
            for contents in ('{"targets": []}', '{"targets": ["phii"]}', '[]', '{'):
                config.write_text(contents)
                with patch.object(pipeline, 'run') as run, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        pipeline.main(['--config', str(config)])
                    self.assertEqual(error.exception.code, 2)
                    run.assert_not_called()

    def test_run_dispatches_only_configured_target_and_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'pipeline.jsonc'
            config.write_text('{"targets": [\n// "phi",\n"gamma",\n]}')
            with patch.object(pipeline, 'RESULTS', root), patch.object(pipeline, 'missing_inputs', return_value=[]), patch('module.model_backend.get_backend') as factory, patch('module.run_artifacts.trained_run', return_value=root/'saved'):
                factory.return_value.predict.return_value={'status':'complete','output':str(root/'saved')}
                self.assertEqual(pipeline.run(stage='distribute', config_path=config), 0)
                factory.return_value.predict.assert_called_once()
            result = json.loads((date_directory(root) / 'run_summary.json').read_text())
            self.assertEqual(result['targets'], ['gamma'])
            self.assertEqual(list(result['stages']['distribute']), ['gamma'])


if __name__ == '__main__':
    unittest.main()
