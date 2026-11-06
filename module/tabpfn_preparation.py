"""Prepare raw c-prime regression tables without fitting encoders or estimators."""
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
import hashlib
import json


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, payload):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    temporary.replace(path)


def export_tables(data, numeric, categorical, output):
    """Export aligned rows; row IDs and provenance are not implicit predictors."""
    import numpy as np
    import pandas as pd
    selected = [*numeric, *categorical]
    if not selected or len(selected) != len(set(selected)) or 'target' in selected or 'row_id' in selected:
        raise ValueError('Predictor names must be nonempty, unique and exclude target/row_id')
    required = {'target', 'boring_id', *selected}
    if required-set(data):
        raise ValueError(f'Missing training columns: {sorted(required-set(data))}')
    frame = data.reset_index(drop=True).copy()
    if frame.empty or frame.boring_id.isna().any():
        raise ValueError('Training table must be nonempty with valid borehole IDs')
    target = pd.to_numeric(frame.target, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError('Effective cohesion must be finite, nonnegative and measured in kPa')
    frame['target'] = target
    frame['row_id'] = np.arange(len(frame))
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for name in categorical:
        frame[name] = frame[name].astype('string')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'model_dataset.csv', index=False)
    return dict(rows=len(frame), boreholes=int(frame.boring_id.nunique()), feature_columns=selected,
                numeric_features=numeric, categorical_features=categorical,
                categorical_indices=[selected.index(name) for name in categorical],
                feature_count=len(selected), target_column='target', target_unit='kPa',
                missing_values={name:int(frame[name].isna().sum()) for name in selected},
                row_id_column='row_id',group_column='boring_id',
                metadata_columns=[name for name in frame if name not in selected and name!='target'],
                row_id_is_predictor=False, preprocessing='raw numeric and categorical data; no imputation/encoding fit',
                split_policy='no split generated in stage 02; fit preprocessing on training partitions only in stage 03')


def _promote_exclusions(output):
    source=output/'shared/excluded_records.csv'
    if source.is_file():
        source.replace(output/'excluded_records.csv')


def _cleanup_intermediates(output):
    """Remove only this run's known duplicates after conditions have been saved."""
    for name in ('shared/model_dataset.csv','shared/preparation_summary.json',
                 'n_distribution/source.json','n_distribution/read_summary.json'):
        (output/name).unlink(missing_ok=True)
    shared=output/'shared'
    if shared.is_dir() and not any(shared.iterdir()):
        shared.rmdir()


def prepare_cohesion():
    """Build one learning table, one condition record, and retained N-grid data."""
    import pandas as pd
    from module.paths import TABPFN_TRAINING, RAW, project_path
    from module import c_model_a
    config=c_model_a.load_config()
    numeric,categorical=config['numeric_features'],config['categorical_features']
    stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
    output=TABPFN_TRAINING/'c'/stamp
    output.mkdir(parents=True,exist_ok=False)
    context=dict(schema_version=2,model='tabpfn',target='effective_cohesion_c_prime',
                 created_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),settings=config)
    _write(output/'metadata.json',dict(context,preparation_status='preparing'))
    _write(output/'status.json',dict(status='preparing',model='tabpfn',target='c'))
    try:
        shared=c_model_a.prepare(output=output/'shared')
        context['shared_preparation']=shared
        _promote_exclusions(output)
        data=pd.read_csv(output/'shared/model_dataset.csv',dtype={name:'string' for name in categorical})
        n_source=None
        if 'n_value' in numeric:
            from module.c_n_distribution import materialize,attach
            artifact=materialize(config['n_distribution'],config,output/'n_distribution')
            data=attach(data,artifact)
            n_source=dict(artifact['metadata'])
            # The original file list already exists at the top of this record.
            if 'read_summary' in n_source:
                n_source['read_summary']={k:v for k,v in n_source['read_summary'].items() if k!='chunks'}
        context['n_distribution_source']=n_source
        manifest=export_tables(data,numeric,categorical,output)
        config_files=['config/model.json','config/validation.json','config/spatial.json',
                      'config/c/prediction.json',config['model_config'],config['n_distribution_config']]
        inputs={str(name):_sha256(project_path(name)) for name in dict.fromkeys(config_files)}
        xml_inputs={str(path.relative_to(RAW)):_sha256(path) for path in sorted((RAW/'soiltest/strength').rglob('*.xml'))}
        packages={}
        for name in ('numpy','pandas','tabpfn'):
            try:packages[name]=version(name)
            except PackageNotFoundError:packages[name]=None
        retained=['model_dataset.csv','excluded_records.csv']
        if n_source is not None:retained.append('n_distribution/n_value.npy')
        manifest.update(context,preparation_status=shared['status'],
                        config_sha256=inputs,source_xml_sha256=xml_inputs,packages=packages,
                        output_sha256={name:_sha256(output/name) for name in retained},
                        target_policy='observed CU-bar/CD effective cohesion only; no pseudo-targets',
                        n_policy='fixed-depth saved N map sampled by XY; not matched to each sample depth',
                        evaluation_scope='upstream_N_holdout_independence_not_verified; do not claim fully independent accuracy',
                        data_reduction='none',model_weights_downloaded=False,models_fitted=False)
        _write(output/'metadata.json',manifest)
        _cleanup_intermediates(output)
        status=dict(status=shared['status'],model='tabpfn',target='c',rows=len(data),
                    output=str(output),dataset=str(output/'model_dataset.csv'),metadata=str(output/'metadata.json'))
        _write(output/'status.json',status)
        return status
    except Exception as error:
        # Failure evidence is retained; never clean failed or older output directories.
        _promote_exclusions(output)
        failure=f'{type(error).__name__}: {error}'
        _write(output/'metadata.json',dict(context,preparation_status='failed',error=failure,
                                          intermediate_files_preserved=True))
        _write(output/'status.json',dict(status='failed',model='tabpfn',target='c',error=failure))
        raise
