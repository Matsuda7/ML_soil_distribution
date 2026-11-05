"""Reuse a completed phi run's N grid; never train an upstream model for c."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from module.paths import RESULTS, project_path
from module.gamma_config import grid_shape
from module.gamma_prediction import grid_indices


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_source(options, current_run=None):
    explicit = current_run or options.get('run_directory')
    if explicit:
        run = project_path(explicit)
    else:
        candidates = []
        if not RESULTS.is_dir():
            raise FileNotFoundError('No results directory; run phi first')
        for day in RESULTS.iterdir():
            if not day.is_dir() or len(day.name) != 8 or not day.name.isdigit():
                continue
            runs = [day / 'phi', *sorted((day / 'archive').glob('phi_*'))]
            for candidate in runs:
                summary = candidate / 'run_summary.json'
                if summary.is_file() and read_json(summary).get('status') == 'complete':
                    candidates.append((summary.stat().st_mtime_ns, str(candidate), candidate))
        if not candidates:
            raise FileNotFoundError('No completed phi/N distribution found. Run phi first.')
        run = max(candidates)[2]
    if read_json(run / 'run_summary.json').get('status') != 'complete':
        raise ValueError(f'N distribution is not complete: {run}')
    config = read_json(run / 'config.json')
    chunks = sorted((run / 'grid/chunks').glob('N_value_10m_*.csv.gz'))
    if not chunks and (run / 'grid/N_value_10m.csv.gz').is_file():
        chunks = [run / 'grid/N_value_10m.csv.gz']
    if not chunks:
        raise FileNotFoundError(f'Completed phi run has no N prediction CSVs: {run}')
    return run.resolve(), config, chunks


def materialize(options, config, output, current_run=None):
    run, source, chunks = select_source(options, current_run)
    spatial = config['spatial']
    source_spatial = source.get('spatial', source)
    for key in ('grid_crs', 'grid_spacing_m', 'property_mask_bounds'):
        if source_spatial[key] != spatial[key]:
            raise ValueError(f'N distribution {key} differs from c: {run}')
    depth = float(source['prediction_depth_m'])
    if not np.isfinite(depth) or depth < 0:
        raise ValueError('Invalid source N distribution depth')
    output.mkdir(parents=True, exist_ok=False)
    shape = grid_shape(spatial)
    grid = np.lib.format.open_memmap(output / 'n_value.npy', mode='w+', dtype=np.float32, shape=shape)
    grid[:] = np.nan
    records = []
    for path in chunks:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        count = 0
        for frame in pd.read_csv(path, usecols=['x', 'y', 'predicted_n_value'], chunksize=100000):
            xi, yi, inside = grid_indices(frame, spatial, shape)
            values = frame.predicted_n_value.to_numpy(dtype=float)
            if not inside.all() or not np.isfinite(values).all() or ((values < 0) | (values > 50)).any():
                raise ValueError(f'Invalid N grid coordinates/values: {path}')
            flat = yi * shape[1] + xi
            if len(np.unique(flat)) != len(flat) or np.isfinite(grid[yi, xi]).any():
                raise ValueError(f'Duplicate N grid coordinates: {path}')
            grid[yi, xi] = values
            count += len(frame)
        records.append(dict(path=str(path), sha256=digest.hexdigest(), rows=count))
    grid.flush()
    metadata = dict(source_run=str(run), source_depth_m=depth, sampling='nearest_grid_cell_xy',
                    depth_policy='same fixed-depth N map for all c sample depths',
                    selection='current_phi_run' if current_run else ('explicit' if options.get('run_directory') else 'latest_completed_summary_mtime'),
                    evaluation_scope='conditional_on_precomputed_N_distribution; upstream_holdout_independence_not_verified',
                    source_config=source, chunks=records)
    (output / 'source.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
    return dict(grid=grid, spatial=spatial, metadata=metadata)


def attach(frame, artifact):
    """Nearest regular cell, with no fill across missing cells or outside the map."""
    result = frame.copy()
    bounds = artifact['spatial']['property_mask_bounds']
    step = float(artifact['spatial']['grid_spacing_m'])
    xy = result[['x', 'y']].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite c sample coordinates')
    inside = ((xy[:, 0] >= bounds['xmin']) & (xy[:, 0] <= bounds['xmax']) &
              (xy[:, 1] >= bounds['ymin']) & (xy[:, 1] <= bounds['ymax']))
    if not inside.all():
        raise ValueError(f'{int((~inside).sum())} c samples outside N distribution; no rows dropped')
    ij = np.floor((xy - [bounds['xmin'], bounds['ymin']]) / step + .5).astype(int)
    values = artifact['grid'][ij[:, 1], ij[:, 0]]
    if not np.isfinite(values).all():
        raise ValueError('N distribution has missing cells at c sample locations; no imputation or rows dropped')
    result['n_value'] = values
    result['n_distribution_depth_m'] = artifact['metadata']['source_depth_m']
    result['n_grid_distance_m'] = np.linalg.norm(xy - (ij * step + [bounds['xmin'], bounds['ymin']]), axis=1)
    return result
