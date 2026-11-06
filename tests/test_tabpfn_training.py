"""Stage-03 test definitions only; deliberately not executed during implementation."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.tabpfn_regression import options_from,check_capacity
from module.tabpfn_training import select_dataset,load_dataset,build_preprocessor,feature_frame
from module.tabpfn_preparation import _sha256


class TabPFNTrainingTests(unittest.TestCase):
    def test_limits_do_not_subsample(self):
        with self.assertRaisesRegex(ValueError,'No subsampling'):
            check_capacity(1001,10,{'model_version':'v2'},{'device':'cpu'})

    def test_invalid_runtime_settings_are_rejected(self):
        for value in (0,-1,1.5,True,'8'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                options_from({'tabpfn':{'cpu_threads':value}})

    def test_current_preparation_has_priority_over_latest_or_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=select_dataset({'dataset_directory':'unused'},prepared_data={'status':'prepared','output':tmp})
            self.assertEqual(path,Path(tmp).resolve())

    def test_dataset_hash_change_stops_before_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'status.json').write_text('{"status":"prepared"}')
            (root/'model_dataset.csv').write_text('changed')
            (root/'metadata.json').write_text(json.dumps(dict(model='tabpfn',target='effective_cohesion_c_prime',
                output_sha256={'model_dataset.csv':'invalid'})))
            with self.assertRaisesRegex(ValueError,'changed'):
                load_dataset(root)

    def test_unknown_categories_do_not_fit_on_validation(self):
        metadata={'numeric_features':['depth'],'categorical_features':['symbol'],'feature_columns':['depth','symbol']}
        pre=build_preprocessor(metadata)
        pre.fit(feature_frame(pd.DataFrame({'depth':[1.,2.],'symbol':['sand','clay']}),metadata))
        result=pre.transform(feature_frame(pd.DataFrame({'depth':[3.],'symbol':['new']}),metadata))
        self.assertEqual(result[0,1],-1)
        self.assertNotIn('new',pre.named_transformers_['categorical'].categories_[0])

    def test_holdout_labels_are_never_fit_and_final_refit_is_separate(self):
        from module.tabpfn_training import train_cohesion
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'data';directory.mkdir();run=Path(tmp)/'result';run.mkdir()
            frame=pd.DataFrame(dict(row_id=range(6),boring_id=list('abcdef'),x=0.,y=0.,depth=1.,target=np.arange(6.)+1))
            metadata=dict(numeric_features=['depth'],categorical_features=[],feature_columns=['depth'],
                          settings={'model_settings':{'training_min_depth_m':0,'training_max_depth_m':10}})
            validation=dict(mode='borehole',test_size=.2,validation_size=.2,random_state=42)
            fitted=[]
            class Model:
                def fit(self,x,y):fitted.append(np.asarray(y).tolist());return self
                def predict(self,x):return np.ones(len(x))
                def save_fit_state(self,path):Path(path).write_text('fake state')
            settings=dict(training_min_depth_m=0,training_max_depth_m=10)
            def dump(pre,path):Path(path).write_text('fake preprocessor')
            with patch('module.configuration.load_model_settings',return_value=settings), \
                 patch('module.tabpfn_training.select_dataset',return_value=directory), \
                 patch('module.tabpfn_training.load_dataset',return_value=(frame,metadata)), \
                 patch('module.tabpfn_training.read_json',return_value={'model_config':'unused'}), \
                 patch('module.xgb_common.load_model_config',return_value=(['depth'],[])), \
                 patch('module.validation.load_config',return_value=validation), \
                 patch('module.validation.split_data',return_value=tuple(frame.iloc[i:i+2].copy() for i in (0,2,4))), \
                 patch('module.tabpfn_regression.runtime',return_value={'model_path':'unused','cpu_threads':1}), \
                 patch('module.tabpfn_regression.check_capacity'), \
                 patch('module.tabpfn_regression.build_regressor',side_effect=lambda *a:Model()), \
                 patch('module.tabpfn_regression.thread_limits',side_effect=lambda *a:nullcontext()), \
                 patch('module.output_layout.create_target_directory',return_value=run), \
                 patch('module.tabpfn_training._sha256',return_value='test'), \
                 patch('importlib.metadata.version',return_value='test'), \
                 patch('module.tabpfn_training._plots'), \
                 patch('module.model_reporting.frequency_report'), \
                 patch('joblib.dump',side_effect=dump):
                result=train_cohesion()
            self.assertEqual(fitted,[[1.,2.],[1.,2.,3.,4.,5.,6.]])
            self.assertEqual(result['status'],'trained')
            self.assertTrue((run/'model/model.tabpfn_fit').is_file())
            self.assertTrue((run/'evaluation/borehole/test_predictions.csv').is_file())
            table=pd.read_csv(run/'evaluation/borehole/metrics.csv')
            self.assertEqual(set(table.prediction_policy),{'raw','nonnegative'})
