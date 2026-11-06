"""Data-export test definitions only. Not executed during implementation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.tabpfn_preparation import export_tables, _promote_exclusions, _cleanup_intermediates
from module.backends import tabpfn


class TabPFNPreparationTests(unittest.TestCase):
    def test_missing_and_categories_preserve_aligned_rows(self):
        data = pd.DataFrame(dict(target=[0., 10., 20.], boring_id=['a','a','b'],
                                 slope=[1., np.nan, np.inf], symbol=['sand',None,'clay']))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            metadata=export_tables(data,['slope'],['symbol'],output)
            table=pd.read_csv(output/'model_dataset.csv',dtype={'symbol':'string'})
            self.assertEqual(table.row_id.tolist(),[0,1,2])
            self.assertEqual(table.target.tolist(),[0.,10.,20.])
            self.assertEqual(table.slope.isna().sum(),2)
            self.assertEqual(table.symbol.isna().sum(),1)
            self.assertEqual({p.name for p in output.iterdir()},{'model_dataset.csv'})
            self.assertEqual(metadata['categorical_indices'],[1])
            self.assertEqual(metadata['feature_columns'],['slope','symbol'])
            self.assertNotIn('boring_id',metadata['feature_columns'])
            self.assertNotIn('row_id',metadata['feature_columns'])

    def test_invalid_target_is_rejected_without_dropping_rows(self):
        for target in (-1.,np.nan,np.inf):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                data=pd.DataFrame(dict(target=[target],boring_id=['a'],depth=[1.]))
                with self.assertRaises(ValueError):
                    export_tables(data,['depth'],[],tmp)

    def test_target_cannot_be_selected_as_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_tables(pd.DataFrame(),['target'],[],tmp)

    def test_unsupported_targets_stop_before_preparation(self):
        with patch('module.tabpfn_preparation.prepare_cohesion') as prepare:
            with self.assertRaises(NotImplementedError):tabpfn.prepare(['c','phi'])
            prepare.assert_not_called()

    def test_backend_only_prepares_and_does_not_fallback(self):
        with patch('module.tabpfn_preparation.prepare_cohesion',return_value={'status':'prepared'}) as prepare:
            self.assertEqual(tabpfn.prepare(['c']),{'c':{'status':'prepared'}})
            prepare.assert_called_once()
        with patch('module.tabpfn_training.train_cohesion',return_value={'status':'trained'}) as train:
            self.assertEqual(tabpfn.train('c'),{'status':'trained'})
            train.assert_called_once_with(prepared_data=None)
        with patch('module.tabpfn_prediction.predict_cohesion',return_value={'status':'complete'}) as predict:
            self.assertEqual(tabpfn.predict('c','unused'),{'status':'complete'})
            predict.assert_called_once_with('unused',n_distribution_run=None)

    def test_cleanup_keeps_conditions_exclusions_n_grid_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'shared').mkdir();(output/'n_distribution').mkdir()
            for name in ('model_dataset.csv','metadata.json','status.json','shared/model_dataset.csv',
                         'shared/preparation_summary.json','shared/excluded_records.csv',
                         'n_distribution/source.json','n_distribution/read_summary.json',
                         'n_distribution/n_value.npy','n_distribution/validation_error.json'):
                (output/name).write_text('test')
            _promote_exclusions(output);_cleanup_intermediates(output)
            self.assertFalse((output/'shared').exists())
            self.assertTrue((output/'excluded_records.csv').is_file())
            self.assertTrue((output/'metadata.json').is_file())
            self.assertTrue((output/'n_distribution/n_value.npy').is_file())
            self.assertTrue((output/'n_distribution/validation_error.json').is_file())
            self.assertFalse((output/'n_distribution/source.json').exists())
            self.assertFalse((output/'n_distribution/read_summary.json').exists())
