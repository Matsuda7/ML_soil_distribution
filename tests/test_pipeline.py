from module.output_layout import date_directory
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module import collection, pipeline
from module.xgb_common import borehole_train_validation_test_split, build_preprocessor, prepare_feature_frame, build_regressor


class CollectionTests(unittest.TestCase):
    def test_import_is_hardlink_and_repeated_import_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.xml'
            target = root / 'raw/target.xml'
            source.write_text('<root/>')
            with patch.object(collection, 'ROOT', root):
                self.assertEqual(collection.import_file(source, target)['status'], 'imported')
                self.assertTrue(source.samefile(target))
                self.assertEqual(collection.import_file(source, target)['status'], 'existing')
                other = root / 'other.xml'
                other.write_text('<different/>')
                with self.assertRaises(ValueError):
                    collection.import_file(other, target)
                self.assertEqual(target.read_text(), '<root/>')


class ModelingTests(unittest.TestCase):
    def test_group_split_can_use_gamma_target_without_hole_leakage(self):
        frame = pd.DataFrame({'hole': np.repeat(np.arange(20), 3), 'gamma': np.arange(60)})
        parts = borehole_train_validation_test_split(frame, target_column='gamma', id_column='hole')
        groups = [set(part.hole) for part in parts]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum(map(len, parts)), len(frame))

    def test_preprocessor_handles_unseen_category_and_model_fits(self):
        train = pd.DataFrame({'x': [1., 2., np.nan, 4.], 'soil': ['sand', 'clay', 'sand', 'clay']})
        transform = build_preprocessor(['x'], ['soil'])
        x = transform.fit_transform(prepare_feature_frame(train, ['x'], ['soil']))
        model = build_regressor({'n_estimators': 2, 'max_depth': 1, 'n_jobs': 1})
        model.fit(x, [10., 20., 15., 30.])
        unseen = pd.DataFrame({'x': [np.nan], 'soil': ['unseen']})
        pred = model.predict(transform.transform(prepare_feature_frame(unseen, ['x'], ['soil'])))
        self.assertTrue(np.isfinite(pred).all())


class PipelineTests(unittest.TestCase):
    """Orchestration tests use hooks only; no model training in these cases."""
    def run_case(self, stages, targets, backend):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'pipeline.jsonc'
            config.write_text(json.dumps(dict(stages=stages,targets=targets)))
            with patch('module.model_backend.selected_model',return_value='xgboost'), \
                 patch('module.model_backend.get_backend',return_value=backend), \
                 patch.object(pipeline,'RESULTS',root), \
                 patch.object(pipeline,'missing_inputs',return_value=[]), \
                 patch('module.run_artifacts.trained_run',return_value=root/'saved'):
                code=pipeline.run(config_path=config)
            report=json.loads((date_directory(root)/'run_summary.json').read_text())
            return code,report

    def test_step_switches_are_strict(self):
        self.assertEqual(pipeline.resolve_stages({'stages':{'01_collect':False,'02_prepare':False}}),('train','distribute'))
        with self.assertRaises(ValueError):pipeline.resolve_stages({'stages':{'03_train':'false'}})
        with self.assertRaises(ValueError):pipeline.resolve_stages({'stages':{'wrong':True}})

    def test_prediction_only_never_calls_training_or_preparation(self):
        from unittest.mock import Mock
        backend=Mock()
        backend.predict.return_value={'status':'complete','output':'saved'}
        code,report=self.run_case(dict(zip(pipeline.STAGE_KEYS.values(),[False,False,False,True])),['phi'],backend)
        self.assertEqual(code,0);backend.prepare.assert_not_called();backend.train.assert_not_called()
        backend.predict.assert_called_once()
        self.assertEqual(report['enabled_stages'],['04_distribute'])

    def test_phi_distribution_precedes_c_training(self):
        from unittest.mock import Mock
        events=[];backend=Mock()
        def train(target,**kwargs):
            events.append('train_'+target)
            if target=='c':self.assertEqual(kwargs['n_distribution_run'],'phi_saved')
            return {'status':'trained','output':target+'_saved'}
        def predict(target,path,**kwargs):
            events.append('predict_'+target)
            return {'status':'complete','output':target+'_saved'}
        backend.train.side_effect=train;backend.predict.side_effect=predict
        code,_=self.run_case({'01_collect':False,'02_prepare':False},['c','phi'],backend)
        self.assertEqual(code,0)
        self.assertEqual(events,['train_phi','predict_phi','train_c','predict_c'])

    def test_failed_phi_never_falls_back_for_c(self):
        from unittest.mock import Mock
        backend=Mock();backend.train.side_effect=RuntimeError('failed phi')
        code,report=self.run_case({'01_collect':False,'02_prepare':False},['phi','c'],backend)
        self.assertEqual(code,2);backend.train.assert_called_once();backend.predict.assert_not_called()
        self.assertEqual(report['stages']['train']['c']['status'],'blocked')



if __name__ == '__main__':
    unittest.main()
