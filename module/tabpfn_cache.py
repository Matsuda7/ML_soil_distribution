"""Ephemeral prediction cache; saved estimators and preprocessing stay immutable."""
from time import perf_counter


def fit_cached_model(model_class, original_params, x, y):
    """Only the engine mode changes; no row reduction or outer preprocessing fit."""
    params = original_params.copy()
    if type(params.get('random_state')) is not int:
        raise ValueError('An immutable integer random_state is required')
    params['fit_mode'] = 'fit_with_cache'
    model = model_class(**params)
    model.fit(x, y)
    return model


def prediction_model(run, model, transform, metadata, metrics, environment, mode):
    report = dict(fit_mode=mode, cache_build_seconds=0., cache_preprocessing_seconds=0.)
    if mode == 'low_memory':
        if model.fit_mode != 'low_memory':
            raise ValueError('Saved model is not low_memory; no implicit engine conversion')
        return model, report
    if mode != 'fit_with_cache':
        raise ValueError('Invalid prediction fit mode')
    if environment['tabpfn_version'] != '2.2.1':
        raise ValueError('Production prediction cache is verified against the 2.2.1 API only')
    import numpy as np
    import pandas as pd
    from module.paths import project_path
    from module.run_artifacts import read_json
    from module.tabpfn_preparation import _sha256
    from module.tabpfn_training import feature_frame
    from module.tabpfn_regression import thread_limits
    prepared = read_json(run/'training_data.json')
    path = project_path(prepared['directory'])/'model_dataset.csv'
    expected = prepared['dataset_sha256']
    if expected != metadata['output_sha256']['model_dataset.csv'] or _sha256(path) != expected:
        raise ValueError('Full stage-03 training dataset is missing or changed; cache not built')
    if metadata['feature_columns'] != metadata['numeric_features'] + metadata['categorical_features']:
        raise ValueError('Invalid saved feature order')
    started = perf_counter()
    data = pd.read_csv(path, dtype={name:'string' for name in metadata['categorical_features'] + ['boring_id']})
    if len(data) != metrics['final_fit_rows'] or not data.row_id.is_unique or data.row_id.isna().any():
        raise ValueError('Cache requires every original final-fit training row in its original order')
    if not np.isfinite(data.target.to_numpy(dtype=float)).all():
        raise ValueError('Invalid saved targets')
    params = model.get_params(deep=False)
    if params['fit_mode'] != 'low_memory' or params['memory_saving_mode'] is not True:
        raise ValueError('Expected saved low_memory / memory_saving_mode=True estimator')
    print(f'TabPFN cache: building from all {len(data):,} saved training rows', flush=True)
    with thread_limits(environment):
        x = transform.transform(feature_frame(data, metadata))
        report['cache_preprocessing_seconds'] = perf_counter()-started
        started = perf_counter()
        try:
            cached = fit_cached_model(type(model), params, x, data.target.to_numpy())
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():
                raise RuntimeError('Cache construction OOM; use prediction_fit_mode=low_memory or request more RAM. No fallback or subsampling performed.') from error
            raise
    report.update(cache_build_seconds=perf_counter()-started, cache_training_rows=len(data),
                  cache_training_dataset_sha256=expected,
                  cache_policy='ephemeral; full saved training rows; saved outer transform only; no model overwrite')
    print(f"TabPFN cache: ready in {report['cache_build_seconds']:.2f} s", flush=True)
    return cached, report
