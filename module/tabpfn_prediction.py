"""Stage 04: restore the saved TabPFN/preprocessor and stream the c grid."""
from decimal import Decimal,InvalidOperation
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import json
import math
import zipfile
from module.paths import project_path
from module.run_artifacts import read_json,write_json,archive_grid,training_record
from module.tabpfn_preparation import _sha256


def prediction_config(run):
    from module.configuration import load_model_settings,load_prediction_settings
    from module.tabpfn_regression import options_from
    saved=read_json(run/'training_config.json')
    if saved.get('estimator')!='tabpfn':raise ValueError('Saved training config is not TabPFN')
    current=load_prediction_settings('config/c/prediction.json')
    spatial=saved['spatial']
    for key in ('grid_crs','grid_spacing_m','property_mask_bounds'):
        if spatial[key]!=current[key]:raise ValueError(f'{key} differs from saved training context')
    for key in ('grid_input','dem_input','property_mask'):
        if project_path(spatial[key]).resolve()!=project_path(current[key]).resolve():
            raise ValueError(f'{key} differs from saved training context')
    saved_options=options_from({'tabpfn':saved['tabpfn']})
    overrides=load_model_settings().get('tabpfn',{})
    # Model family, estimator count and checkpoint are fixed by stage 03.
    for key in ('device','cpu_threads','predict_batch_size'):
        if key in overrides:saved_options[key]=overrides[key]
    options=options_from({'tabpfn':saved_options})
    for key in ('prediction_depth_m','observed_depth_tolerance_m','input_chunk_size','display_spacing_m','write_text_matrix'):
        saved[key]=current[key]
    if type(saved['input_chunk_size']) is not int or saved['input_chunk_size']<1:
        raise ValueError('input_chunk_size must be a positive integer')
    ratio=saved['display_spacing_m']/float(spatial['grid_spacing_m'])
    if not math.isfinite(ratio) or ratio<1 or not math.isclose(ratio,round(ratio)):
        raise ValueError('display_spacing_m must be a positive multiple of grid spacing')
    if type(saved['write_text_matrix']) is not bool:raise ValueError('write_text_matrix must be boolean')
    saved.update(tabpfn=options,output_dir=str(run),retain_raw_predictions=True,plot_model_label='TabPFN',prediction_id=uuid4().hex)
    return saved,options


def prediction_frame(data,metadata,preprocessor):
    """Use saved category vocabulary; never fit on prediction coordinates."""
    import numpy as np
    import pandas as pd
    frame=data[metadata['feature_columns']].copy()
    for name in metadata['numeric_features']:
        frame[name]=pd.to_numeric(frame[name],errors='coerce').replace([np.inf,-np.inf],np.nan)
    encoder=preprocessor.named_transformers_.get('categorical')
    for index,name in enumerate(metadata['categorical_features']):
        values=frame[name].astype('string')
        # Existing CSVs can store integer geographic codes as e.g. 1101.0.
        # Match only to unique, already-trained representations, never create categories.
        if name in ('jshis_jcode','ser') and encoder is not None:
            learned={str(value) for value in encoder.categories_[index] if isinstance(value,str)}
            aliases={};ambiguous=set()
            for label in learned:
                try:key=Decimal(label)
                except InvalidOperation:continue
                if not key.is_finite():continue
                if key in aliases and aliases[key]!=label:ambiguous.add(key)
                aliases[key]=label
            def known_representation(value):
                if pd.isna(value) or value in learned:return value
                try:key=Decimal(value)
                except InvalidOperation:return value
                if not key.is_finite() or key in ambiguous:return value
                return aliases.get(key,value)
            values=values.map(known_representation)
        frame[name]=values.astype(object).where(values.notna(),np.nan)
    return frame


def load_assets(run,config,options):
    import numpy as np
    import joblib
    from module.tabpfn_regression import runtime,check_capacity,thread_limits
    artifacts=read_json(run/'model/artifacts.json')
    for filename,key in (('model.tabpfn_fit','model_sha256'),('preprocessor.joblib','preprocessor_sha256')):
        if _sha256(run/'model'/filename)!=artifacts[key]:raise ValueError(f'Saved artifact changed: {filename}')
    metrics=read_json(run/'model/metrics.json')
    prepared=read_json(run/'training_data.json')
    metadata=prepared['metadata']
    if metrics['numeric_features']!=metadata['numeric_features'] or metrics['categorical_features']!=metadata['categorical_features']:
        raise ValueError('Saved model features and stage-02 metadata differ')
    if config['numeric_features']!=metadata['numeric_features'] or config['categorical_features']!=metadata['categorical_features']:
        raise ValueError('Saved training config features differ from model metadata')
    environment=runtime(options)
    if _sha256(environment['model_path'])!=artifacts['checkpoint_sha256']:
        raise ValueError('Base checkpoint differs from the one used in stage 03')
    check_capacity(metrics['final_fit_rows'],len(metadata['feature_columns']),options,environment)
    with zipfile.ZipFile(run/'model/model.tabpfn_fit') as archive:
        params=json.loads(archive.read('init_params.json'))
    original=params.get('model_path')
    if not isinstance(original,str) or not Path(original).is_file() or Path(original).resolve()!=Path(environment['model_path']).resolve():
        raise ValueError('Restore the original local checkpoint path recorded in the fitted state; no automatic download')
    transform=joblib.load(run/'model/preprocessor.joblib')
    if list(transform.feature_names_in_)!=metadata['feature_columns']:
        raise ValueError('Saved preprocessor input columns differ from stage 02')
    artifact=None
    if 'n_value' in metadata['numeric_features']:
        directory=project_path(prepared['directory']);path=directory/'n_distribution/n_value.npy'
        expected=metadata['output_sha256'].get('n_distribution/n_value.npy')
        if not expected or _sha256(path)!=expected:raise ValueError('Saved stage-02 N grid is missing or changed')
        grid=np.load(path,mmap_mode='r')
        from module.gamma_config import grid_shape
        if grid.shape!=grid_shape(config['spatial']):raise ValueError('Saved N grid shape differs from target grid')
        artifact=dict(grid=grid,spatial=config['spatial'],metadata=metadata['n_distribution_source'])
    # Official restoration initializes stored context, not a new fit.
    from tabpfn import TabPFNRegressor
    with thread_limits(environment):
        model=TabPFNRegressor.load_from_fit_state(run/'model/model.tabpfn_fit',device=environment['device'])
    # Models saved before the training-depth range was included remain usable.
    if 'training_depth_range_m' not in metrics:
        import pandas as pd
        dataset=project_path(prepared['directory'])/'model_dataset.csv'
        if _sha256(dataset)!=prepared['dataset_sha256']:raise ValueError('Training data changed; cannot restore depth range')
        depths=pd.read_csv(dataset,usecols=['depth']).depth
        metrics['training_depth_range_m']=[float(depths.min()),float(depths.max())]
    return model,transform,metadata,metrics,artifact,environment


class BatchedPredictor:
    def __init__(self,model,size,progress_path):
        self.model=model;self.size=size;self.path=progress_path
        self.rows=0;self.seconds=0.;self.batches=0

    def _progress(self,rows,seconds):
        self.rows+=rows;self.seconds+=seconds;self.batches+=1
        write_json(self.path,dict(status='predicting',model_prediction_rows=self.rows,batches=self.batches,
                   inference_seconds=self.seconds,points_per_second=self.rows/max(self.seconds,1e-9),
                   note='progress only; not a resume checkpoint'))
        if self.batches==1 or self.batches%20==0:
            print(f'TabPFN inference: {self.rows:,} points, {self.rows/max(self.seconds,1e-9):.2f} points/s',flush=True)

    def predict(self,values):
        from module.tabpfn_regression import predict_batches
        return predict_batches(self.model,values,self.size,progress=self._progress)


def predict_cohesion(run,n_distribution_run=None):
    run=project_path(run)
    status,_=training_record(run,'c')
    if status.get('status')!='trained' or status.get('estimator')!='tabpfn' or status.get('target')!='c':
        raise ValueError('A completed TabPFN c stage-03 run is required')
    if n_distribution_run is not None:
        raise NotImplementedError('TabPFN 04 reuses the N distribution frozen in stage 02; no replacement by a different phi run')
    config,options=prediction_config(run)
    model,transform,metadata,metrics,n_artifact,environment=load_assets(run,config,options)
    # Required local inputs are checked before an old distribution is archived.
    for key in ('grid_input','dem_input','property_mask'):
        if not project_path(config['spatial'][key]).is_file():raise FileNotFoundError(config['spatial'][key])
    archive_grid(run)
    write_json(run/'config.json',config)
    write_json(run/'run_summary.json',dict(status='predicting',estimator='tabpfn',output=str(run)))
    write_json(run/'prediction_environment.json',dict(environment,batch_size=options['predict_batch_size']))
    predictor=BatchedPredictor(model,options['predict_batch_size'],run/'grid/prediction_progress.json')
    started=perf_counter()
    try:
        from module.tabpfn_regression import thread_limits
        from module.c_prediction import predict_grid
        with thread_limits(environment):
            report=predict_grid(config,predictor,transform,metrics,run/'grid',n_artifact=n_artifact,
                                feature_builder=lambda frame:prediction_frame(frame,metadata,transform))
        report.update(estimator='tabpfn',output=str(run),batch_size=options['predict_batch_size'],
                      model_prediction_rows=predictor.rows,inference_seconds=predictor.seconds,
                      points_per_second=predictor.rows/max(predictor.seconds,1e-9),total_seconds=perf_counter()-started)
        if n_artifact is not None:
            report['n_distribution_source']=n_artifact['metadata']['source_run']
            report['n_distribution_depth_m']=n_artifact['metadata']['source_depth_m']
        write_json(run/'grid/prediction_summary.json',report)
        write_json(run/'grid/prediction_progress.json',dict(report,note='progress only; not a resume checkpoint'))
        write_json(run/'run_summary.json',report)
        return report
    except Exception as error:
        failure=dict(status='failed',estimator='tabpfn',output=str(run),error=f'{type(error).__name__}: {error}',
                     model_prediction_rows=predictor.rows,total_seconds=perf_counter()-started)
        write_json(run/'run_summary.json',failure)
        if (run/'grid').is_dir():write_json(run/'grid/prediction_progress.json',failure)
        raise
