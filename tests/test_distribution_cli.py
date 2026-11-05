"""CLI definitions only; execution is deliberately omitted for this task."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from module import full_distribution as full


class DistributionCLITests(unittest.TestCase):
    def test_plan_never_initializes_analysis(self):
        with patch.object(full,'initialize',side_effect=AssertionError('analysis started')), contextlib.redirect_stdout(io.StringIO()) as output:
            result=full.main(['--comparison-run','unused','--output','unused','--stage','xgboost','--plan'])
        self.assertEqual(result,0)
        self.assertIn('"analysis_executed": false',output.getvalue())

    def test_xgboost_dispatch_does_not_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(full,'initialize',return_value=({},{})), patch.object(full,'recompute_xgboost') as predict, patch.object(full,'prepare') as prepare:
                result=full.main(['--comparison-run','unused','--output',tmp,'--stage','xgboost'])
            self.assertEqual(result,0)
            predict.assert_called_once()
            prepare.assert_not_called()
