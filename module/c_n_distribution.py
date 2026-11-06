"""Reuse a completed phi run's N grid; never train an upstream model for c."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from module.paths import RESULTS, project_path
from module.gamma_config import grid_shape


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


def inspect_grid_batch(frame, spatial, shape, existing, row_offset=0, crop_outside=False):
    """Inspect one read batch without changing coordinates, N values, or row count."""
    numeric = frame[['x','y','predicted_n_value']].apply(pd.to_numeric, errors='coerce')
    xy = numeric[['x','y']].to_numpy(dtype=float)
    values = numeric.predicted_n_value.to_numpy(dtype=float)
    bounds = spatial['property_mask_bounds'];step=float(spatial['grid_spacing_m'])
    finite_xy = np.isfinite(xy).all(axis=1)
    with np.errstate(over='ignore', invalid='ignore'):
        rounded = np.rint((xy-[bounds['xmin'],bounds['ymin']])/step)
    inside = (finite_xy & (rounded[:,0]>=0) & (rounded[:,0]<shape[1]) &
              (rounded[:,1]>=0) & (rounded[:,1]<shape[0]))
    # Cast only known in-bounds indices; malformed/huge coordinates cannot overflow int64.
    xi = np.zeros(len(frame),dtype=np.int64);yi=xi.copy()
    xi[inside]=rounded[inside,0].astype(np.int64);yi[inside]=rounded[inside,1].astype(np.int64)
    aligned = np.zeros(len(frame),dtype=bool)
    aligned[inside] = (np.isclose(xy[inside,0],bounds['xmin']+xi[inside]*step,rtol=0,atol=1e-4) &
                       np.isclose(xy[inside,1],bounds['ymin']+yi[inside]*step,rtol=0,atol=1e-4))
    repeated=np.zeros(len(frame),dtype=bool);previous=repeated.copy()
    flat=yi[inside]*shape[1]+xi[inside]
    repeated[inside]=pd.Series(flat).duplicated(keep=False).to_numpy()
    previous[inside]=np.isfinite(existing[yi[inside],xi[inside]])
    finite_n=np.isfinite(values)
    masks = {
        'non_finite_or_non_numeric_coordinates':~finite_xy,
        'outside_grid_index_range':finite_xy & ~inside,
        'coordinates_not_aligned_to_grid':inside & ~aligned,
        'non_finite_or_non_numeric_n':~finite_n,
        'n_below_zero':finite_n & (values<0),
        'n_above_50':finite_n & (values>50),
        'duplicate_coordinates_in_batch':repeated,
        'duplicate_coordinates_in_previous_batches':previous,
    }
    def scalar(value):
        if value is None or value is pd.NA: return None
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,float) and not np.isfinite(value):return str(value)
        return value
    def example(index):
        row=frame.iloc[index]
        return dict(csv_row_number=int(row_offset+index+2),
                    **{key:scalar(row[key]) for key in ('x','y','predicted_n_value')})
    ranges={}
    for name in numeric:
        data=numeric[name].to_numpy(dtype=float);data=data[np.isfinite(data)]
        ranges[name]=dict(min=float(data.min()),max=float(data.max())) if len(data) else dict(min=None,max=None)
    ignored = finite_xy & ~inside
    ignored_count = int(ignored.sum()) if crop_outside else 0
    if crop_outside:
        # The source CSV can legitimately cover a larger area than the requested map.
        # Unknown/nonfinite coordinates remain errors; only known outside cells are cropped.
        masks['outside_grid_index_range'] = np.zeros(len(frame),dtype=bool)
        for name in ('non_finite_or_non_numeric_n','n_below_zero','n_above_50'):
            masks[name] = masks[name] & ~ignored
    counts={name:int(mask.sum()) for name,mask in masks.items()}
    diagnostics=dict(
        scope='first failing read batch only; later batches have not been checked',
        csv_row_start=int(row_offset+2),csv_row_end=int(row_offset+len(frame)+1),batch_rows=len(frame),
        expected=dict(crs=spatial.get('grid_crs'),bounds=bounds,grid_spacing_m=step,
                      shape=list(shape),n_range=[0,50],coordinate_tolerance_m=1e-4),
        observed_finite_ranges=ranges,counts=counts,
        ignored_counts={'outside_requested_grid':ignored_count},
        ignored_examples=[example(int(i)) for i in np.flatnonzero(ignored)[:5]] if crop_outside else [],
        examples={name:[example(int(i)) for i in np.flatnonzero(mask)[:5]] for name,mask in masks.items() if mask.any()})
    if crop_outside:
        return xi[inside],yi[inside],values[inside],diagnostics
    return xi,yi,values,diagnostics


def raise_grid_diagnostics(path, output, diagnostics):
    """Persist bounded error examples and include the exact files in the traceback."""
    report=Path(output)/'validation_error.json'
    examples=Path(output)/'validation_error_rows.csv'
    diagnostics=dict(source_file=str(path),**diagnostics)
    report.write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    rows=[]
    for reason,items in diagnostics['examples'].items():
        rows.extend(dict(reason=reason,**item) for item in items)
    pd.DataFrame(rows,columns=['reason','csv_row_number','x','y','predicted_n_value']).to_csv(examples,index=False)
    counts={name:value for name,value in diagnostics['counts'].items() if value}
    raise ValueError(
        f'Invalid N grid: {path}; failed checks={counts}; '
        f'expected bounds={diagnostics["expected"]["bounds"]}, spacing={diagnostics["expected"]["grid_spacing_m"]} m, N=[0,50]; '
        f'observed finite ranges (this batch)={diagnostics["observed_finite_ranges"]}; '
        f'details={report}; example rows={examples}. Only the first failing batch was checked; no values corrected or c training rows removed. Outside-domain N cells may have been cropped (see ignored_counts).')


def materialize(options, config, output, current_run=None):
    run, source, chunks = select_source(options, current_run)
    spatial = config['spatial']
    source_spatial = source.get('spatial', source)
    for key in ('grid_crs', 'grid_spacing_m', 'property_mask_bounds'):
        if source_spatial[key] != spatial[key]:
            raise ValueError(f'N distribution {key} mismatch: source={source_spatial[key]!r}, expected={spatial[key]!r}; source run={run}')
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
        count = 0;used = 0;outside = 0
        for frame in pd.read_csv(path, usecols=['x', 'y', 'predicted_n_value'], chunksize=100000):
            xi,yi,values,diagnostics=inspect_grid_batch(frame,spatial,shape,grid,row_offset=count,crop_outside=True)
            if any(diagnostics['counts'].values()):
                raise_grid_diagnostics(path,output,diagnostics)
            grid[yi, xi] = values
            count += len(frame);used += len(values)
            outside += diagnostics['ignored_counts']['outside_requested_grid']
        records.append(dict(path=str(path), sha256=digest.hexdigest(), rows=count,
                            used_rows=used,outside_grid_rows_ignored=outside))
        if outside:
            print(f'N distribution {path.name}: source rows={count:,}, used={used:,}, outside requested grid={outside:,} (cropped)',flush=True)
    grid.flush()
    read_summary = dict(source_run=str(run),requested_bounds=spatial['property_mask_bounds'],
                        source_rows=sum(item['rows'] for item in records),
                        used_rows=sum(item['used_rows'] for item in records),
                        outside_grid_rows_ignored=sum(item['outside_grid_rows_ignored'] for item in records),
                        policy='crop known outside-domain N cells; preserve all c samples; validate retained N cells',chunks=records)
    (output/'read_summary.json').write_text(json.dumps(read_summary,indent=2)+'\n')
    if read_summary['used_rows']==0:
        raise ValueError(f'N distribution has no cells in the requested grid; details={output / "read_summary.json"}')
    metadata = dict(source_run=str(run), source_depth_m=depth, sampling='nearest_grid_cell_xy',
                    depth_policy='same fixed-depth N map for all c sample depths',
                    selection='current_phi_run' if current_run else ('explicit' if options.get('run_directory') else 'latest_completed_summary_mtime'),
                    evaluation_scope='conditional_on_precomputed_N_distribution; upstream_holdout_independence_not_verified',
                    source_config=source, chunks=records,read_summary=read_summary)
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
