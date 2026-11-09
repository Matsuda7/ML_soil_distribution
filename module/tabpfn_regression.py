"""Optional TabPFN regressor adapter. No torch/tabpfn import until stage 03."""
from contextlib import contextmanager
from importlib.metadata import version, PackageNotFoundError
import os
from pathlib import Path
import socket
import warnings
from module.paths import project_path

FAMILIES = {'v2': '2.2.1', 'v3.5-fast': '9.2.0'}
DEFAULTS = dict(dataset_directory=None,model_version='v2',
                model_path='data/models/tabpfn/tabpfn-v2-regressor.ckpt',device='auto',
                cpu_threads=None,n_estimators=8,predict_batch_size=256,prediction_fit_mode='low_memory')


def options_from(settings):
    custom=settings.get('tabpfn',{})
    if not isinstance(custom,dict):raise ValueError('model.json tabpfn settings must be an object')
    unknown=set(custom)-set(DEFAULTS)
    if unknown:raise ValueError(f'Unknown tabpfn settings: {sorted(unknown)}')
    options = {**DEFAULTS,**custom}
    if options['model_version'] not in FAMILIES:
        raise ValueError('tabpfn.model_version must be v2 or v3.5-fast')
    if options['device'] not in ('auto','cpu','cuda'):
        raise ValueError('tabpfn.device must be auto, cpu or cuda')
    for key in ('n_estimators','predict_batch_size'):
        if type(options[key]) is not int or options[key]<1:
            raise ValueError(f'tabpfn.{key} must be an integer >= 1')
    value=options['cpu_threads']
    if value is not None and (type(value) is not int or value<1):
        raise ValueError('tabpfn.cpu_threads must be an integer >= 1 or null')
    if options['prediction_fit_mode'] not in ('low_memory','fit_with_cache'):
        raise ValueError('tabpfn.prediction_fit_mode must be low_memory or fit_with_cache')
    path=options['model_path']
    if not isinstance(path,str) or not path.strip() or path=='auto':
        raise ValueError('tabpfn.model_path must name an explicit local checkpoint')
    dataset=options['dataset_directory']
    if dataset is not None and (not isinstance(dataset,str) or not dataset.strip()):
        raise ValueError('tabpfn.dataset_directory must be null or a dataset directory')
    return options


def _cpu_plan(requested):
    try:affinity=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):affinity=None
    available=min(os.cpu_count() or 1,affinity or os.cpu_count() or 1)
    allocations=[]
    raw=os.environ.get('PBS_NP')
    if raw:
        if not raw.isdecimal() or int(raw)<1:raise ValueError('Invalid PBS_NP')
        allocations.append(int(raw))
    nodefile=os.environ.get('PBS_NODEFILE')
    if nodefile:
        hosts={socket.gethostname().split('.')[0].lower(),socket.getfqdn().split('.')[0].lower()}
        slots=sum(line.strip().split('.')[0].lower() in hosts for line in Path(nodefile).read_text().splitlines() if line.strip())
        if not slots:raise ValueError('PBS_NODEFILE has no slots for this host')
        allocations.append(slots)
    for key in ('PBS_NUM_NODES','PBS_NNODES'):
        if not nodefile and os.environ.get(key) not in (None,'','1'):
            raise ValueError('Multi-node PBS jobs require PBS_NODEFILE to determine local allocation')
    if os.environ.get('PBS_JOBID') and not allocations:
        raise ValueError('Cannot determine PBS CPU allocation')
    pbs=min(allocations) if allocations else None
    desired=requested if requested is not None else (pbs or available)
    effective=min(desired,available,pbs or available)
    if desired>effective:warnings.warn(f'CPU threads capped from {desired} to {effective}',RuntimeWarning)
    return dict(cpu_threads_configured=requested,cpu_threads=effective,pbs_cpus=pbs,affinity_cpus=affinity)


def runtime(options):
    expected=FAMILIES[options['model_version']]
    try:installed=version('tabpfn')
    except PackageNotFoundError as error:
        raise RuntimeError(f'This Python needs tabpfn=={expected}; no installation was performed') from error
    if installed!=expected:
        raise RuntimeError(f'{options["model_version"]} requires tabpfn=={expected}; found {installed}')
    checkpoint=project_path(options['model_path']).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Local checkpoint required: {checkpoint}; no automatic download')
    if options['model_version']=='v3.5-fast' and checkpoint.name!='tabpfn-v3.5-fast-20260909.safetensors':
        raise ValueError('Keep the official 3.5-Fast checkpoint filename')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TABPFN_NO_BROWSER']='1'
    os.environ['TABPFN_DISABLE_TELEMETRY']='1'
    import torch
    device=options['device']
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    plan=_cpu_plan(options['cpu_threads'])
    torch.set_num_threads(plan['cpu_threads'])
    return dict(plan,device=device,torch_threads=torch.get_num_threads(),torch_version=torch.__version__,
                cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0) if device=='cuda' else None,
                tabpfn_version=installed,model_version=options['model_version'],model_path=str(checkpoint))


def check_capacity(rows,columns,options,environment):
    if options['model_version']=='v2':
        limit=1000 if environment['device']=='cpu' else 10000
        if rows>limit or columns>500:
            raise ValueError(f'TabPFN v2 capacity exceeded: {rows} rows/{columns} features, '
                             f'{environment["device"]} limit={limit} rows/500 features. No subsampling performed.')
    elif rows>1000000 or columns>20000:
        raise ValueError('TabPFN 3.5-Fast training limit exceeded; no subsampling performed')
    # The official implementation also checks checkpoint-specific CPU/pretraining limits.


def build_regressor(options,environment,seed,categorical_indices):
    from tabpfn import TabPFNRegressor
    kwargs=dict(model_path=environment['model_path'],device=environment['device'],random_state=seed,
                n_estimators=options['n_estimators'],categorical_features_indices=categorical_indices,
                fit_mode='low_memory',memory_saving_mode=True,ignore_pretraining_limits=False)
    if options['model_version']=='v3.5-fast':
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST,n_preprocessing_jobs=1,**kwargs)
    return TabPFNRegressor(n_jobs=1,**kwargs)


@contextmanager
def thread_limits(environment):
    # Runtime BLAS control works even when NumPy was imported by the shared CLI.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=environment['cpu_threads']):
        yield


def predict_batches(model,values,size,progress=None):
    import numpy as np
    result=[]
    from time import perf_counter
    for start in range(0,len(values),size):
        started=perf_counter()
        try:pred=np.asarray(model.predict(values[start:start+size]),dtype=float)
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():
                raise RuntimeError('Prediction OOM: reduce predict_batch_size; no training rows were removed') from error
            raise
        if pred.shape!=(len(values[start:start+size]),) or not np.isfinite(pred).all():
            raise ValueError('Invalid TabPFN predictions')
        result.append(pred)
        if progress is not None:progress(len(pred),perf_counter()-started)
    return np.concatenate(result) if result else np.empty(0)
