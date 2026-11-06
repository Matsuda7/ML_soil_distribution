"""XGBoost hooks: prepare, train/validate, and saved-model distribution."""
from pathlib import Path
import json
from module.paths import project_path
from module.run_artifacts import read_json, write_json, archive_grid, training_record

NAME = 'xgboost'


def prepare(targets):
    from module.preparation import prepare_phi, prepare_strength
    result = {}
    if 'phi' in targets:
        result['phi'] = prepare_phi()
    strength = [target for target in targets if target in ('c','gamma')]
    if strength:
        result.update(prepare_strength(strength))
    return result


def train(target, gamma_kind=None, n_distribution_run=None, config_override=None, prepared_data=None):
    if target == 'phi':
        from module import phi_training, phi_prediction
        from module.output_layout import create_target_directory
        from module.config_snapshot import write_config_snapshot
        run = create_target_directory('phi')
        config = phi_prediction.load_prediction_config(phi_prediction.DEFAULT_CONFIG, run)
        write_config_snapshot(config, run, 'phi')
        write_json(run/'run_summary.json', {'status':'training','output':str(run)})
        try:
            phi_training.main(run/'model')
        except Exception as error:
            write_json(run/'run_summary.json', {'status':'failed','error':str(error)})
            raise
        result = {'status':'trained','output':str(run)}
    elif target == 'c':
        from module.c_model_a import train as train_c
        result = train_c(n_distribution_run)
        run = Path(result['output'])
    elif target == 'gamma':
        from module.gamma import train as train_gamma
        from module.gamma_config import load_config
        config = config_override or load_config()
        if gamma_kind is not None: config['gamma_kind'] = gamma_kind
        result = train_gamma(config)
        run = Path(result['output'])
    else:
        raise ValueError(f'Unknown target: {target}')
    if result['status'] != 'trained':
        raise ValueError(f'Training did not complete: {result}')
    config = read_json(run/'config.json')
    config['estimator'] = NAME
    write_json(run/'training_config.json',config)
    write_json(run/'config.json',config)
    write_json(run/'training_status.json',dict(status='trained',target=target,estimator=NAME,
               gamma_kind=config.get('gamma_kind'),output=str(run)))
    write_json(run/'run_summary.json',dict(result,estimator=NAME))
    return result


def _prediction_config(target, run):
    from module.configuration import load_prediction_settings
    saved = read_json(run/('training_config.json' if (run/'training_config.json').is_file() else 'config.json'))
    if saved.get('estimator','xgboost') != NAME:
        raise ValueError('Saved model belongs to a different estimator')
    current = load_prediction_settings(f'config/{target}/prediction.json')
    spatial = saved.get('spatial',saved)
    for key in ('grid_crs','grid_spacing_m','property_mask_bounds'):
        if spatial[key] != current[key]:
            raise ValueError(f'Prediction {key} differs from saved training context; train with the intended spatial settings')
    for key in ('grid_input','dem_input','property_mask'):
        if project_path(spatial[key]).resolve() != project_path(current[key]).resolve():
            raise ValueError(f'Prediction {key} differs from saved training context')
    size = current.get('input_chunk_size')
    if type(size) is not int or size < 1:
        raise ValueError('input_chunk_size must be a positive integer')
    import math
    ratio = current['display_spacing_m']/float(current['grid_spacing_m'])
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio,round(ratio)):
        raise ValueError('display_spacing_m must be a positive multiple of grid spacing')
    # Feature/soil/model settings come from training. Only explicit prediction knobs can change.
    for key in ('prediction_depth_m','observed_depth_tolerance_m','input_chunk_size','display_spacing_m',
                'write_text_matrix','create_distribution','n_plot_min','n_plot_max',
                'include_outside_dem_boreholes','outside_dem_fill_value','existing_chunks'):
        if key in current: saved[key] = current[key]
    saved['output_dir'] = str(run)
    return saved


def _load_regressor(directory):
    import joblib
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(directory/'model.json')
    return model, joblib.load(directory/'preprocessor.joblib'), read_json(directory/'metrics.json')


def predict(target, run, max_chunks=None, n_distribution_run=None):
    """Stage 04 loads fitted estimators only: never call fit or a training hook."""
    run = project_path(run)
    status,_ = training_record(run,target)
    if status.get('status') != 'trained' or status.get('estimator') != NAME or status.get('target') != target:
        raise ValueError('Stage 04 requires a compatible completed stage-03 model')
    config = _prediction_config(target,run)
    if target == 'phi':
        from module import phi_prediction
        config.update(model_dir=run/'model', output_chunk_dir=run/'grid/chunks',
                      n_value_plot=run/'grid/N_value_10m.png',phi_plot=run/'grid/phi_10m.png',
                      phi_text=run/'grid/phi_10m.txt.gz')
        # Restore path types used by the existing phi prediction implementation.
        for key in ('grid_input','dem_input','dem_cache','property_mask','borehole_data'):
            config[key] = project_path(config[key])
        phi_prediction.load_model_artifacts(config)
        phi_prediction.validate_inputs(config)
        predict = lambda: _predict_phi(config,max_chunks)
    elif target == 'c':
        import numpy as np
        from module.c_prediction import predict_grid
        model, pre, metrics = _load_regressor(run/'model')
        artifact = None
        if 'n_value' in config['numeric_features']:
            artifact = dict(grid=np.load(run/'n_distribution/n_value.npy',mmap_mode='r'),
                            spatial=config['spatial'],metadata=read_json(run/'n_distribution/source.json'))
            current_id = read_json(project_path(n_distribution_run)/'config.json').get('prediction_id') if n_distribution_run else None
            same_source = (n_distribution_run and current_id is not None
                           and Path(artifact['metadata']['source_run']).resolve() == project_path(n_distribution_run).resolve()
                           and artifact['metadata']['source_config'].get('prediction_id') == current_id)
            if n_distribution_run and not same_source:
                from uuid import uuid4
                from module.c_n_distribution import materialize
                artifact = materialize(config['n_distribution'],config,run/'prediction_n_sources'/uuid4().hex,n_distribution_run)
            config['n_distribution_source'] = artifact['metadata']
        predict = lambda: predict_grid(config,model,pre,metrics,run/'grid',n_artifact=artifact)
    elif target == 'gamma':
        from module.gamma_prediction import predict_grid
        kind = config['gamma_kind']
        models = {kind: _load_regressor(run/'models'/kind)}
        artifact = None
        if config['distribution_model'] == 'predicted_n':
            import joblib
            from xgboost import XGBRegressor
            upstream = run/'n_models/grid'
            n_models = []
            for seed in config['n_comparison']['ensemble_seeds']:
                model = XGBRegressor()
                model.load_model(upstream/f'model_seed_{seed}.json');n_models.append(model)
            artifact = (n_models,joblib.load(upstream/'preprocessor.joblib'),read_json(upstream/'metrics.json'))
        predict = lambda: (predict_grid(config,models,run/'grid',max_chunks=max_chunks,n_artifact=artifact)
                           if config.get('create_distribution',True) else {'status':'complete','distribution':'not_requested'})
    else:
        raise ValueError(f'Unknown target: {target}')
    archive_grid(run)
    if not (run/'training_config.json').is_file():
        legacy=read_json(run/'config.json');legacy['estimator']=NAME
        write_json(run/'training_config.json',legacy)
        write_json(run/'training_status.json',status)
    from uuid import uuid4
    config['prediction_id']=uuid4().hex
    write_json(run/'config.json',config)
    write_json(run/'run_summary.json',dict(status='predicting',estimator=NAME,output=str(run)))
    try:
        result = dict(predict(), output=str(run), estimator=NAME)
    except Exception as error:
        write_json(run/'run_summary.json',dict(status='failed',error=str(error),output=str(run),estimator=NAME))
        raise
    write_json(run/'run_summary.json',result)
    return result


def _predict_phi(config,max_chunks):
    from module import phi_prediction
    rows = phi_prediction.predict_grid(config,max_chunks,False)
    if max_chunks is not None:
        return dict(status='incomplete',predicted_points=rows)
    count = phi_prediction.save_phi_matrix(config['output_chunk_dir'],config['phi_text'],config)
    display = phi_prediction.load_display_points(config['output_chunk_dir'],config['display_spacing_m'])
    phi_prediction.save_plots(display,config)
    return dict(status='complete',predicted_points=rows,phi_points=count)
