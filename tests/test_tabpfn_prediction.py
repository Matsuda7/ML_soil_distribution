"""Saved-prediction test definitions only; no tests are run during this edit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from module.tabpfn_prediction import BatchedPredictor,prediction_frame
from module.tabpfn_training import build_preprocessor,feature_frame


class TabPFNPredictionTests(unittest.TestCase):
    def test_batches_use_predict_only_and_report_progress(self):
        calls=[]
        class Model:
            def predict(self,x):calls.append(len(x));return np.ones(len(x))
        with tempfile.TemporaryDirectory() as tmp:
            predictor=BatchedPredictor(Model(),3,Path(tmp)/'progress.json')
            values=predictor.predict(np.zeros((8,2)))
            self.assertEqual(calls,[3,3,2]);self.assertEqual(len(values),8)
            self.assertEqual(predictor.rows,8);self.assertTrue((Path(tmp)/'progress.json').is_file())

    def test_grid_codes_use_only_saved_category_representations(self):
        metadata={'numeric_features':['depth'],'categorical_features':['jshis_jcode'],
                  'feature_columns':['depth','jshis_jcode']}
        pre=build_preprocessor(metadata)
        pre.fit(feature_frame(pd.DataFrame({'depth':[1.,2.],'jshis_jcode':['1101.0','1102.0']}),metadata))
        before=pre.named_transformers_['categorical'].categories_[0].copy()
        frame=prediction_frame(pd.DataFrame({'depth':[3.,4.],'jshis_jcode':[1101,9999]}),metadata,pre)
        encoded=pre.transform(frame)
        self.assertEqual(frame.jshis_jcode.iloc[0],'1101.0')
        self.assertEqual(encoded[1,1],-1)
        np.testing.assert_array_equal(before,pre.named_transformers_['categorical'].categories_[0])

    def test_shared_grid_writer_retains_raw_and_nonnegative_values(self):
        from module.c_prediction import predict_grid
        class Model:
            def predict(self,x):return np.array([-1.,2.,3.,4.])
        class Transform:
            def transform(self,x):return x.to_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'grid.txt'
            source.write_text('0 0 0 0\n10 0 0 0\n0 10 0 0\n10 10 0 0\n')
            spatial=dict(grid_input=str(source),property_mask='unused',grid_spacing_m=10.,property_mask_spacing_m=10.,
                         property_mask_bounds=dict(xmin=0.,xmax=10.,ymin=0.,ymax=10.),grid_crs='EPSG:6675')
            config=dict(spatial=spatial,prediction_depth_m=5.,input_chunk_size=100,display_spacing_m=10.,
                        numeric_features=['depth'],categorical_features=[],write_text_matrix=False,retain_raw_predictions=True)
            with patch('module.c_prediction.read_binary_mask',return_value=np.ones((2,2),dtype=np.uint8)), \
                 patch('module.c_prediction.load_dem'), \
                 patch('module.c_prediction.geographic_context',return_value=(None,None)), \
                 patch('module.c_prediction.enrich',side_effect=lambda points,*args:points.assign(surface_z=10.,elevation=5.)), \
                 patch('module.c_prediction.plot_grid'):
                result=predict_grid(config,Model(),Transform(),{'training_depth_range_m':[0,10]},root/'out',
                                    feature_builder=lambda frame:frame[['depth']])
            self.assertEqual(result['status'],'complete')
            raw=np.load(root/'out/c_effective_raw_kpa.npy')
            bounded=np.load(root/'out/c_effective_kpa.npy')
            self.assertEqual(raw[0,0],-1.);self.assertEqual(bounded[0,0],0.)
            table=pd.read_csv(root/'out/chunks/c_effective_000001.csv.gz')
            self.assertEqual(table.c_effective_raw_kpa.tolist(),[-1.,2.,3.,4.])
            self.assertEqual(table.c_effective_kpa.tolist(),[0.,2.,3.,4.])
