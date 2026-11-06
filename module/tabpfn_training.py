"""Stage 03 for effective cohesion: audit prepared rows, holdout evaluation, final fit."""
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from module.paths import TABPFN_TRAINING,ROOT,project_path
from module.tabpfn_preparation import _sha256
from module.run_artifacts import read_json,write_json


def select_dataset(options,prepared_data=None):
    if prepared_data is not None:
        if prepared_data.get('status')!='prepared':raise ValueError('Current stage 02 did not complete')
        selected=project_path(prepared_data['output'])
    elif options.get('dataset_directory'):
        selected=project_path(options['dataset_directory'])
    else:
        base=TABPFN_TRAINING/'c';candidates=[]
        for path in base.iterdir() if base.is_dir() else []:
            status=path/'status.json'
            if status.is_file() and read_json(status).get('status')=='prepared':
                candidates.append((status.stat().st_mtime_ns,str(path),path))
        if not candidates:raise FileNotFoundError('No completed TabPFN c dataset; run 02_prepare_data.py --targets c')
        selected=max(candidates)[2]
    return selected.resolve()


def load_dataset(directory):
    import numpy as np
    import pandas as pd
    status=read_json(directory/'status.json');meta=read_json(directory/'metadata.json')
    if status.get('status')!='prepared' or meta.get('model')!='tabpfn' or meta.get('target')!='effective_cohesion_c_prime':
        raise ValueError('Stage 03 requires a completed TabPFN effective-cohesion dataset')
    hashes=meta.get('output_sha256',{})
    if 'model_dataset.csv' not in hashes:raise ValueError('Dataset has no content hash; regenerate stage 02')
    for name,expected in hashes.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or _sha256(path)!=expected:
            raise ValueError(f'Prepared output missing or changed: {name}')
    numeric=meta['numeric_features'];categorical=meta['categorical_features'];features=meta['feature_columns']
    if features!=numeric+categorical or len(set(features))!=len(features) or not features:
        raise ValueError('Inconsistent feature order in prepared metadata')
    if {'target','row_id','boring_id'} & set(features):raise ValueError('Target/IDs cannot be predictors')
    frame=pd.read_csv(directory/'model_dataset.csv',dtype={name:'string' for name in categorical+['boring_id']})
    required={'row_id','boring_id','target','x','y','depth',*features}
    if required-set(frame):raise ValueError(f'Missing dataset fields: {sorted(required-set(frame))}')
    if len(frame)!=meta['rows'] or not frame.row_id.is_unique or frame[['row_id','boring_id']].isna().any(axis=None):
        raise ValueError('Invalid row IDs, borehole IDs, or prepared row count')
    if meta.get('target_unit')!='kPa':raise ValueError('Effective cohesion unit must be kPa')
    values=frame[['target','x','y','depth']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame.target<0).any():raise ValueError('Invalid target/coordinates/depth')
    for name in numeric:
        frame[name]=pd.to_numeric(frame[name],errors='raise')
        if np.isinf(frame[name].to_numpy(dtype=float)).any():raise ValueError(f'Infinite predictor: {name}')
    return frame,meta


def feature_frame(data,metadata):
    import numpy as np
    frame=data[metadata['feature_columns']].copy()
    for name in metadata['categorical_features']:
        frame[name]=frame[name].astype(object).where(frame[name].notna(),np.nan)
    return frame


def build_preprocessor(metadata):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    transforms=[('numeric','passthrough',metadata['numeric_features'])]
    if metadata['categorical_features']:
        transforms.append(('categorical',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,
                          encoded_missing_value=float('nan'),dtype=float),metadata['categorical_features']))
    return ColumnTransformer(transforms,remainder='drop',sparse_threshold=0)


def score(actual,predicted):
    import numpy as np
    from module.xgb_common import regression_metrics
    return dict(regression_metrics(actual,predicted),bias=float(np.mean(predicted-np.asarray(actual,dtype=float))))


def _plots(table,directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from module.model_reporting import depth_scatter
    fig,ax=plt.subplots(figsize=(6,5))
    depth_scatter(ax,table.target,table.predicted_c_effective_kpa,table.depth,s=20,alpha=.7)
    high=max(float(table.target.max()),float(table.predicted_c_effective_kpa.max()),1.)
    ax.plot([0,high],[0,high],'k--',lw=1)
    metrics=score(table.target,table.predicted_c_effective_kpa.to_numpy())
    ax.text(.04,.96,'\n'.join(f'{key.upper()} = {metrics[key]:.4f}' for key in ('r2','rmse','mae')),
            transform=ax.transAxes,va='top')
    ax.set(xlabel="Observed c-prime [kPa]",ylabel="Predicted c-prime [kPa]",title='TabPFN held-out test')
    fig.tight_layout();fig.savefig(directory/'observed_vs_predicted.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.hist(table.predicted_raw-table.target,bins=20,alpha=.6,label='raw')
    ax.hist(table.predicted_c_effective_kpa-table.target,bins=20,alpha=.6,label='nonnegative')
    ax.axvline(0,color='black',linestyle='--');ax.legend();ax.set(xlabel='Prediction - observation [kPa]',ylabel='Count')
    fig.tight_layout();fig.savefig(directory/'residual_distribution.png',dpi=180);plt.close(fig)


def train_cohesion(prepared_data=None):
    import numpy as np
    import pandas as pd
    import joblib
    from module.configuration import load_model_settings
    from module.validation import load_config,split_data
    from module.xgb_common import load_model_config
    from module.output_layout import create_target_directory
    from module.tabpfn_regression import options_from,runtime,check_capacity,build_regressor,thread_limits,predict_batches
    settings=load_model_settings();options=options_from(settings)
    directory=select_dataset(options,prepared_data)
    data,metadata=load_dataset(directory)
    # Prepared feature/depth conditions are immutable. New model parameters need no new stage 02.
    current=read_json(project_path('config/c/prediction.json'))
    numeric,categorical=load_model_config(project_path(current['model_config']))
    if numeric!=metadata['numeric_features'] or categorical!=metadata['categorical_features']:
        raise ValueError('Feature settings changed after stage 02; regenerate the learning table')
    for key in ('training_min_depth_m','training_max_depth_m'):
        if settings[key]!=metadata['settings']['model_settings'][key]:
            raise ValueError(f'{key} changed after stage 02; regenerate the learning table')
    validation=load_config()
    if validation['mode']!='borehole':raise ValueError('TabPFN c stage 03 requires borehole holdout to prevent hole leakage')
    parts=split_data(data,validation,target_column='target')
    if min(map(len,parts))<2:raise ValueError('Each train/validation/test partition needs at least two samples')
    environment=runtime(options)
    check_capacity(len(data),len(metadata['feature_columns']),options,environment)
    seed=validation['random_state'];categorical_indices=list(range(len(numeric),len(numeric)+len(categorical)))
    # Fail before archiving existing results if data, checkpoint, or environment is unavailable.
    run=create_target_directory('c');evaluation=run/'evaluation'/validation['mode'];evaluation.mkdir(parents=True)
    model_dir=run/'model';model_dir.mkdir()
    config=dict(metadata['settings'],estimator='tabpfn',tabpfn=options,validation=validation,
                numeric_features=numeric,categorical_features=categorical,training_dataset=str(directory/'model_dataset.csv'),
                prepared_dataset_directory=str(directory),output_dir=str(run),model_settings=settings)
    write_json(run/'config.json',config);write_json(run/'training_config.json',config)
    write_json(run/'training_status.json',dict(status='training',target='c',estimator='tabpfn'))
    write_json(run/'run_summary.json',dict(status='training',target='c',estimator='tabpfn',output=str(run)))
    try:
        try:git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.PIPE).strip()
        except (OSError,subprocess.CalledProcessError):git_sha=None
        try:git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,stderr=subprocess.PIPE).strip()
        except (OSError,subprocess.CalledProcessError):git_status=None
        from importlib.metadata import version
        environment['packages']={name:version(name) for name in ('numpy','pandas','scikit-learn','joblib','tabpfn','torch')}
        environment.update(git_sha=git_sha,git_status=git_status,python=platform.python_version(),platform=platform.platform(),
                           source_files_sha256={name:_sha256(ROOT/name) for name in (
                               'module/tabpfn_training.py','module/tabpfn_regression.py','module/backends/tabpfn.py',
                               'module/validation.py','module/xgb_common.py','module/pipeline.py')},
                           checkpoint_sha256=_sha256(environment['model_path']),
                           started_at=datetime.now(timezone.utc).isoformat())
        write_json(run/'environment.json',environment)
        write_json(run/'training_data.json',dict(directory=str(directory),metadata=metadata,
                   dataset_sha256=_sha256(directory/'model_dataset.csv')))
        splits=pd.concat([part[['row_id','boring_id','x','y','depth']].assign(split=name)
                          for name,part in zip(('train','validation','test'),parts)],ignore_index=True)
        splits.to_csv(evaluation/'split.csv',index=False)
        transform=build_preprocessor(metadata)
        started=perf_counter();xt=transform.fit_transform(feature_frame(parts[0],metadata))
        validation_x=transform.transform(feature_frame(parts[1],metadata));test_x=transform.transform(feature_frame(parts[2],metadata))
        preprocessing_seconds=perf_counter()-started
        model=build_regressor(options,environment,seed,categorical_indices)
        with thread_limits(environment):
            started=perf_counter();model.fit(xt,parts[0].target.to_numpy());fit_seconds=perf_counter()-started
            scores=[];predictions=[]
            for name,part,x in [('validation',parts[1],validation_x),('test',parts[2],test_x)]:
                started=perf_counter();raw=predict_batches(model,x,options['predict_batch_size']);seconds=perf_counter()-started
                corrected=np.maximum(raw,0.)
                table=part.copy();table['split']=name;table['predicted_raw']=raw
                table['predicted_c_effective_kpa']=corrected;table['residual_raw']=raw-part.target.to_numpy()
                table['residual']=corrected-part.target.to_numpy();table.to_csv(evaluation/f'{name}_predictions.csv',index=False)
                predictions.append(table)
                for policy,values in [('raw',raw),('nonnegative',corrected)]:
                    scores.append(dict(split=name,prediction_policy=policy,**score(part.target,values),
                                       train_rows=len(parts[0]),rows=len(part),fit_seconds=fit_seconds,
                                       preprocessing_seconds=preprocessing_seconds,predict_seconds=seconds))
            model.save_fit_state(evaluation/'evaluation_model.tabpfn_fit')
        joblib.dump(transform,evaluation/'preprocessor.joblib')
        pd.DataFrame(scores).to_csv(evaluation/'metrics.csv',index=False)
        write_json(evaluation/'metrics.json',dict(metrics=scores,validation=validation,
                   split_method='borehole_train_validation_test_holdout; not spatial CV or OOF',
                   upstream_N_independence='not_verified',parameters=options))
        _plots(predictions[-1],evaluation)
        from module.model_reporting import frequency_report
        frequency_report(parts[0].target,evaluation,"Effective cohesion c-prime [kPa]")
        # Final refit has its own preprocessing and context. It does not change held-out metrics.
        del model,transform
        final_transform=build_preprocessor(metadata)
        started=perf_counter();all_x=final_transform.fit_transform(feature_frame(data,metadata))
        final_preprocessing_seconds=perf_counter()-started
        final=build_regressor(options,environment,seed,categorical_indices)
        with thread_limits(environment):
            started=perf_counter();final.fit(all_x,data.target.to_numpy());final_fit_seconds=perf_counter()-started
            final.save_fit_state(model_dir/'model.tabpfn_fit')
        joblib.dump(final_transform,model_dir/'preprocessor.joblib')
        frequency_report(data.target,model_dir,"Effective cohesion c-prime [kPa]")
        write_json(model_dir/'metrics.json',dict(final_fit_rows=len(data),final_fit_holes=int(data.boring_id.nunique()),
                   training_depth_range_m=[float(data.depth.min()),float(data.depth.max())],
                   numeric_features=numeric,categorical_features=categorical,categorical_indices=categorical_indices,
                   target_unit='kPa',parameters=options,final_fit_seconds=final_fit_seconds,
                   preprocessing_seconds=final_preprocessing_seconds,
                   evaluation_reference=str(evaluation.relative_to(run)/'metrics.json'),
                   evaluation_scope='conditional_on_saved_N_distribution; upstream_holdout_independence_not_verified',
                   feature_importance='not_computed; XGBoost gain is not available for TabPFN',
                   prediction_lower_bound=0.,raw_predictions_retained=True))
        write_json(model_dir/'artifacts.json',dict(model_sha256=_sha256(model_dir/'model.tabpfn_fit'),
                   preprocessor_sha256=_sha256(model_dir/'preprocessor.joblib'),checkpoint_sha256=environment['checkpoint_sha256']))
        status=dict(status='trained',target='c',estimator='tabpfn',output=str(run),
                    prepared_dataset=str(directory),completed_at=datetime.now(timezone.utc).isoformat())
        write_json(run/'training_status.json',status);write_json(run/'run_summary.json',status)
        return status
    except Exception as error:
        status=dict(status='failed',target='c',estimator='tabpfn',output=str(run),error=f'{type(error).__name__}: {error}')
        write_json(run/'training_status.json',status);write_json(run/'run_summary.json',status)
        raise
