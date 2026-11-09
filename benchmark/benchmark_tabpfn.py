#!/usr/bin/env python3
"""Read-only production inputs; isolated, sequential TabPFN 2.2.1 experiments."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be an integer >= 1')
    return value


def validate(config):
    keys = {'run', 'cpu_threads', 'prediction_rows', 'batch_sizes', 'fit_modes',
            'repeats', 'warmup_rows', 'seed', 'total_prediction_rows', 'depths_m',
            'timeout_seconds', 'atol', 'rtol'}
    if set(config) != keys:
        raise ValueError(f'Config fields missing={keys-set(config)}, unknown={set(config)-keys}')
    for name in ('prediction_rows', 'repeats', 'warmup_rows', 'total_prediction_rows', 'timeout_seconds'):
        positive_int(config[name], name)
    if config['cpu_threads'] is not None:
        positive_int(config['cpu_threads'], 'cpu_threads')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(config['run'], str) or not config['run']:
        raise ValueError('run must be the stage-03 directory')
    batches = config['batch_sizes']
    if not isinstance(batches, list) or not batches or len(set(batches)) != len(batches):
        raise ValueError('batch_sizes must be a nonempty unique integer list')
    for size in batches:
        positive_int(size, 'batch_size')
    if 256 not in batches:
        raise ValueError('batch_sizes must include the production reference 256')
    modes = config['fit_modes']
    if not isinstance(modes, list) or not modes or len(set(modes)) != len(modes) or set(modes)-{'low_memory', 'fit_with_cache'}:
        raise ValueError('Invalid fit_modes')
    if 'low_memory' not in modes:
        raise ValueError('low_memory reference is required')
    for name in ('atol', 'rtol'):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and >= 0')
    depths = config['depths_m']
    if depths is not None and (not isinstance(depths, list) or not depths or
            any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in depths) or
            len(set(depths)) != len(depths)):
        raise ValueError('depths_m must be null or unique finite nonnegative depths')
    return config


def schedule(config):
    rng = random.Random(config['seed'])
    result = []
    for repeat in range(1, config['repeats'] + 1):
        block = [(mode, size) for mode in config['fit_modes'] for size in config['batch_sizes']]
        rng.shuffle(block)
        result.extend(dict(fit_mode=mode, batch_size=size, repeat=repeat) for mode, size in block)
    return result


def extrapolate(rows, seconds, load, build, total):
    rate = rows / seconds
    return dict(points_per_second=rate,
                estimated_inference_seconds_20m=total / rate,
                estimated_total_seconds_20m=load + build + total / rate)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configure_cpu(config):
    # No numerical-library imports before these environment settings.
    from module.tabpfn_regression import _cpu_plan
    plan = _cpu_plan(config['cpu_threads'])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(plan['cpu_threads'])
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TABPFN_NO_BROWSER'] = '1'
    os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
    return plan


def assets(config):
    from module.paths import project_path
    from module.tabpfn_prediction import prediction_config, load_assets
    from module.run_artifacts import training_record
    run = project_path(config['run']).resolve()
    status, _ = training_record(run, 'c')
    if status.get('estimator') != 'tabpfn' or status.get('status') != 'trained':
        raise ValueError('A trained TabPFN c run is required')
    settings, options = prediction_config(run)
    if options['model_version'] != 'v2':
        raise ValueError('This benchmark is specific to TabPFN 2.2.1 / v2')
    options.update(device='cpu', cpu_threads=config['cpu_threads'])
    model, transform, metadata, metrics, n_grid, env = load_assets(run, settings, options)
    if model.fit_mode != 'low_memory' or model.memory_saving_mode is not True:
        raise ValueError('Reference must be the saved low_memory / memory_saving_mode=True model')
    return run, settings, model, transform, metadata, metrics, n_grid, env


def reservoir_update(old_points, old_keys, points, keys, count):
    """Uniform random priorities; O(chunk + sample) storage, not all coordinates."""
    import numpy as np
    all_points = np.concatenate([old_points, points])
    all_keys = np.concatenate([old_keys, keys])
    keep = np.argpartition(all_keys, count-1)[:count] if len(all_keys) > count else np.arange(len(all_keys))
    return all_points[keep], all_keys[keep]


def prepare(config, output):
    import numpy as np
    import pandas as pd
    from module.paths import project_path
    from module.gamma_config import grid_shape
    from module.gamma_prediction import grid_indices
    from module.raster import read_binary_mask
    from module.gamma_features import load_dem, geographic_context, enrich
    from module.tabpfn_prediction import prediction_frame
    from module.tabpfn_regression import thread_limits
    from module.c_n_distribution import attach
    started = perf_counter()
    run, settings, model, transform, metadata, metrics, n_grid, env = assets(config)
    saved_model_parameters = model.get_params(deep=False)
    del model
    spatial = settings['spatial']
    shape = grid_shape(spatial)
    mask = read_binary_mask(project_path(spatial['property_mask']), shape)
    seen = np.zeros(shape, dtype=bool)
    rng = np.random.default_rng(config['seed'])
    points = np.empty((0, 2)); keys = np.empty(0); total = 0
    reader = pd.read_csv(project_path(spatial['grid_input']), sep=r'\s+', header=None,
                         names=['x', 'y', 'legacy_z', 'legacy_bedrock'], usecols=[0, 1],
                         chunksize=settings['input_chunk_size'])
    for chunk in reader:
        chunk = chunk.apply(pd.to_numeric, errors='coerce')
        xi, yi, inside = grid_indices(chunk, spatial, shape)
        active = np.zeros(len(chunk), dtype=bool)
        active[inside] = mask[yi[inside], xi[inside]] != 0
        xi, yi = xi[active], yi[active]
        flat = yi * shape[1] + xi
        if len(np.unique(flat)) != len(flat) or seen[yi, xi].any():
            raise ValueError('Duplicate active grid cells, matching production validation')
        seen[yi, xi] = True
        selected = chunk.loc[active, ['x', 'y']].to_numpy(dtype=float)
        total += len(selected)
        points, keys = reservoir_update(points, keys, selected, rng.random(len(selected)), config['prediction_rows'])
    missing = int(np.count_nonzero((mask != 0) & ~seen))
    if missing or len(points) != config['prediction_rows']:
        raise ValueError(f'Incomplete grid or insufficient sample: missing={missing}, sample={len(points)}')
    order = np.argsort(keys)
    sample = pd.DataFrame(points[order], columns=['x', 'y'])
    depths = config['depths_m'] or [settings['prediction_depth_m']]
    # Select actual cell-depth pairs, balancing requested depths without changing training.
    sample['depth'] = np.resize(np.asarray(depths, dtype=float), len(sample))
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    # DEM cache must also be private: never create/update production caches.
    settings['spatial'] = dict(spatial, dem_cache=str(output/'dem_cache.npy'))
    with thread_limits(env):
        dem = load_dem(settings)
        geology, jshis = geographic_context(settings)
        frame = enrich(sample, settings, dem, geology, jshis)
        frame['sample_elevation'] = frame.surface_z - frame.depth
        if n_grid is not None:
            frame = attach(frame, n_grid)
        raw_features = prediction_frame(frame, metadata, transform)
        encoded = np.asarray(transform.transform(raw_features), dtype=float)
    np.save(output/'input.npy', encoded)
    sample.to_csv(output/'sample_coordinates.csv', index=False)
    raw_features.to_csv(output/'sample_features.csv', index=False)
    from importlib.metadata import version
    manifest = dict(run=str(run), saved_model_parameters=saved_model_parameters,
                    packages={name:version(name) for name in ('tabpfn','torch','numpy','pandas','scikit-learn','joblib')},
                    prediction_rows=len(sample), active_xy_cells=total,
                    depths_m=depths, feature_columns=metadata['feature_columns'], environment=env,
                    preparation_seconds=perf_counter()-started,
                    input_sha256=sha256(output/'input.npy'),
                    model_sha256=sha256(run/'model/model.tabpfn_fit'),
                    preprocessor_sha256=sha256(run/'model/preprocessor.joblib'),
                    checkpoint_sha256=sha256(env['model_path']),
                    n_distribution_source=metadata.get('n_distribution_source'),
                    note='Fixed-depth production grid; optional depths are explicitly selected cell-depth pairs')
    write_json(output/'sample_manifest.json', manifest)


def differences(pred, reference, atol, rtol):
    import numpy as np
    delta = pred - reference
    return dict(max_abs_prediction_diff=float(np.max(np.abs(delta))),
                mean_abs_prediction_diff=float(np.mean(np.abs(delta))),
                rmse_prediction_diff=float(np.sqrt(np.mean(delta**2))),
                bias_prediction_diff=float(np.mean(delta)),
                max_abs_corrected_diff=float(np.max(np.abs(np.maximum(pred, 0)-np.maximum(reference, 0)))),
                predictions_close=bool(np.allclose(pred, reference, atol=atol, rtol=rtol)))


from module.tabpfn_cache import fit_cached_model


def peak_memory_mb():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024*1024 if sys.platform == 'darwin' else 1024)


def trial(config, output, job):
    import gc
    import numpy as np
    from module.paths import project_path
    from module.tabpfn_regression import predict_batches, thread_limits
    from module.tabpfn_training import feature_frame
    from module.tabpfn_preparation import _sha256
    import pandas as pd
    started = perf_counter()
    run, settings, model, transform, meta, metrics, _, env = assets(config)
    load_seconds = perf_counter()-started
    manifest = read_json(output/'sample_manifest.json')
    for path, key in [(run/'model/model.tabpfn_fit', 'model_sha256'),
                      (run/'model/preprocessor.joblib', 'preprocessor_sha256'),
                      (Path(env['model_path']), 'checkpoint_sha256'),
                      (output/'input.npy', 'input_sha256')]:
        if sha256(path) != manifest[key]:
            raise ValueError(f'Input changed during benchmark: {path}')
    values = np.load(output/'input.npy', mmap_mode='r')
    if len(values) != config['prediction_rows']:
        raise ValueError('Sample row count changed')
    build_seconds = 0.; rebuild_preprocessing_seconds = 0.
    with thread_limits(env):
        if job['fit_mode'] == 'fit_with_cache':
            prepared = read_json(run/'training_data.json')
            table = project_path(prepared['directory'])/'model_dataset.csv'
            if _sha256(table) != prepared['dataset_sha256'] or _sha256(table) != meta['output_sha256']['model_dataset.csv']:
                raise ValueError('Full saved training data changed')
            tick = perf_counter()
            data = pd.read_csv(table, dtype={name: 'string' for name in meta['categorical_features'] + ['boring_id']})
            if len(data) != metrics['final_fit_rows'] or not data.row_id.is_unique:
                raise ValueError('Final training rows do not match stage 03')
            x = transform.transform(feature_frame(data, meta))
            y = data.target.to_numpy()
            rebuild_preprocessing_seconds = perf_counter()-tick
            params = model.get_params(deep=False).copy()
            model_cls = type(model)
            del model
            gc.collect()
            tick = perf_counter()
            model = fit_cached_model(model_cls, params, x, y)
            build_seconds = perf_counter()-tick
        build_peak = peak_memory_mb()
        tick = perf_counter()
        predict_batches(model, values[:min(config['warmup_rows'], len(values))], job['batch_size'])
        warmup_seconds = perf_counter()-tick
        tick = perf_counter()
        predicted = predict_batches(model, values, job['batch_size'])
        seconds = perf_counter()-tick
    peak = peak_memory_mb()
    np.save(output/(job['id']+'_predictions.npy'), predicted)
    if job['id'] == 'reference':
        diff = differences(predicted, predicted, config['atol'], config['rtol'])
    else:
        reference = np.load(output/'reference_predictions.npy')
        diff = differences(predicted, reference, config['atol'], config['rtol'])
    return dict(job, cpu_threads=env['torch_threads'], prediction_rows=len(values),
                model_load_seconds=load_seconds, cache_build_seconds=build_seconds,
                rebuild_preprocessing_seconds=rebuild_preprocessing_seconds, warmup_seconds=warmup_seconds,
                peak_memory_mb=peak, cache_build_peak_memory_mb=build_peak,
                **extrapolate(len(values), seconds, load_seconds, build_seconds + rebuild_preprocessing_seconds,
                              config['total_prediction_rows']),
                inference_seconds=seconds, **diff,
                status='success' if diff['predictions_close'] else 'prediction_mismatch')


def child_main(args):
    output = Path(args.output)
    config = read_json(output/'benchmark_config.json')
    configure_cpu(config)
    if args.prepare:
        prepare(config, output)
        return
    job = read_json(output/'jobs.json')[args.worker]
    try:
        row = trial(config, output, job)
    except Exception as error:
        traceback.print_exc()
        row = dict(job, status='failed', error=f'{type(error).__name__}: {error}', peak_memory_mb=peak_memory_mb())
    write_json(output/(job['id']+'_result.json'), row)
    if row['status'] == 'failed':
        raise SystemExit(1)


def run_child(args, output, name, timeout):
    with (output/(name+'.log')).open('x') as log:
        try:
            code = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                   '--output', str(output), *args], stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode
            return code, None
        except subprocess.TimeoutExpired:
            return -1, 'timeout'


def save_csv(output, rows):
    required = ['fit_mode', 'batch_size', 'cpu_threads', 'prediction_rows',
                'model_load_seconds', 'cache_build_seconds', 'inference_seconds',
                'points_per_second', 'peak_memory_mb', 'max_abs_prediction_diff', 'status']
    fields = list(dict.fromkeys(required + [key for row in rows for key in row]))
    with (output/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--prediction-rows', type=int)
    parser.add_argument('--batch-sizes', type=int, nargs='+')
    parser.add_argument('--run')
    parser.add_argument('--output', help=argparse.SUPPRESS)
    parser.add_argument('--prepare', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare or args.worker is not None:
        child_main(args)
        return
    if not os.environ.get('PBS_JOBID'):
        parser.error('Run the benchmark inside a PBS job; local use is limited to --help and lightweight tests')
    config = read_json(args.config)
    for arg, field in [('cpu_threads', 'cpu_threads'), ('prediction_rows', 'prediction_rows'),
                       ('batch_sizes', 'batch_sizes'), ('run', 'run')]:
        if getattr(args, arg) is not None:
            config[field] = getattr(args, arg)
    validate(config)
    # Safe planning only: no torch import, downloads, or inference in coordinator.
    plan = configure_cpu(config)
    base = ROOT/'results/benchmarks/tabpfn'
    output = base/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'benchmark_config.json', config)
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    write_json(output/'benchmark_environment.json', dict(plan, git_sha=git_sha, python=sys.executable,
               source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__).resolve(), ROOT/'module/tabpfn_prediction.py', ROOT/'module/tabpfn_regression.py']},
               pbs_jobid=os.environ.get('PBS_JOBID'), hostname=os.uname().nodename))
    print(f'Benchmark output: {output}', flush=True)
    code, reason = run_child(['--prepare'], output, 'prepare', config['timeout_seconds'])
    if code:
        write_json(output/'benchmark_status.json', dict(status='failed', stage='prepare', reason=reason, exit_code=code))
        raise SystemExit(f'Sample preparation failed; see {output}/prepare.log')
    jobs = [dict(id='reference', fit_mode='low_memory', batch_size=256, repeat=0)]
    jobs += [dict(job, id=f'trial_{i:03d}') for i, job in enumerate(schedule(config), start=1)]
    write_json(output/'jobs.json', jobs)
    rows = []
    for index, job in enumerate(jobs):
        print(f"Starting {job['id']}: {job['fit_mode']} batch={job['batch_size']}", flush=True)
        code, reason = run_child(['--worker', str(index)], output, job['id'], config['timeout_seconds'])
        path = output/(job['id']+'_result.json')
        row = read_json(path) if path.is_file() else dict(job, status='failed', error=reason or f'Worker exit {code}; see log (possible OS/PBS memory kill)')
        if code and row.get('status') == 'success':
            row.update(status='failed', error=f'Worker exit {code}')
        rows.append(row); save_csv(output, rows)
        print(f"{job['id']}: {row['status']}", flush=True)
        if index == 0 and row['status'] != 'success':
            break
    complete = len(rows) == len(jobs) and all(row['status'] == 'success' for row in rows)
    write_json(output/'benchmark_status.json', dict(status='complete' if complete else 'incomplete', conditions_finished=len(rows), conditions_planned=len(jobs)))
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
