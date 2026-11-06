"""Saved-model selection and prediction history; no learning is performed here."""
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4
from module.paths import RESULTS, project_path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')
    temporary.replace(path)


def target_name(target, gamma_kind=None):
    if target == 'gamma':
        if gamma_kind is None:
            from module.gamma_config import load_config
            gamma_kind = load_config()['gamma_kind']
        return 'gamma_' + gamma_kind
    return target


def training_record(run,target):
    """Use stage-03 markers, or validate a complete legacy all-data model."""
    run=Path(run)
    marker=run/'training_status.json'
    if marker.is_file():
        return read_json(marker),marker
    summary=run/'run_summary.json';snapshot=run/'config.json'
    if not summary.is_file() or not snapshot.is_file() or read_json(summary).get('status')!='complete':
        raise ValueError(f'No compatible stage-03 model record: {run}')
    config=read_json(snapshot)
    if target=='phi':
        directory=run/'model'
        if not (directory/'metrics.json').is_file(): directory=directory/'artifacts'
        model_file='xgb_spt_model.json';pre_file='xgb_preprocessor.joblib'
    else:
        directory=run/'model' if target=='c' else run/'models'/config.get('gamma_kind','wet')
        model_file='model.json';pre_file='preprocessor.joblib'
    if not all((directory/name).is_file() for name in (model_file,pre_file,'metrics.json')):
        raise ValueError(f'No complete saved final model: {run}')
    metrics=read_json(directory/'metrics.json')
    if not isinstance(metrics.get('final_fit_rows'),int) or metrics['final_fit_rows']<1:
        raise ValueError(f'Saved artifact is not an all-data final model: {run}')
    return dict(status='trained',target=target,estimator=config.get('estimator','xgboost'),
                gamma_kind=config.get('gamma_kind'),legacy_validated=True),summary


def trained_run(target, estimator, explicit=None, gamma_kind=None):
    name = target_name(target, gamma_kind)
    if explicit:
        candidates = [project_path(explicit)]
    else:
        candidates = []
        if RESULTS.is_dir():
            for day in RESULTS.iterdir():
                if day.is_dir() and len(day.name)==8 and day.name.isdigit():
                    candidates.extend([day/name, *sorted((day/'archive').glob(name+'_*'))])
    available = []
    for run in candidates:
        try: status,state=training_record(run,target)
        except (OSError,ValueError,KeyError):
            if explicit: raise
            continue
        if status.get('status') != 'trained' or status.get('estimator') != estimator or status.get('target') != target:
            if explicit: raise ValueError(f'Incomplete or incompatible saved model: {run}')
            continue
        if target == 'gamma' and 'gamma_'+str(status.get('gamma_kind')) != name:
            if explicit: raise ValueError(f'Saved gamma kind does not match {name}')
            continue
        available.append((state.stat().st_mtime_ns, str(run), run))
    if not available:
        raise FileNotFoundError(f'No completed saved model for {name}/{estimator}; enable 03_train or set reuse_runs')
    return max(available)[2].resolve()


def archive_grid(run):
    """Preserve the previous prediction and its resolved settings before a rerun."""
    grid = run/'grid'
    if grid.exists():
        destination = run/'prediction_archive'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
        destination.mkdir(parents=True)
        grid.rename(destination/'grid')
        for name in ('config.json', 'run_summary.json', 'prediction_environment.json'):
            path = run/name
            if path.is_file():
                import shutil
                shutil.copy2(path,destination/name)
