"""Numbered analysis workflow with shared data and model-specific stage hooks."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import traceback
from module.paths import CONFIG, RESULTS, project_path
from module.collection import collect, missing_inputs
from module.xgb_common import load_jsonc_config

STAGES = ('collect','prepare','train','distribute')
STAGE_KEYS = dict(zip(STAGES,('01_collect','02_prepare','03_train','04_distribute')))
TARGETS = ('c','phi','gamma')


def resolve_targets(config, override=None):
    selected = override if override is not None else config.get('targets',list(TARGETS))
    if not isinstance(selected,(list,tuple)) or not selected or any(not isinstance(t,str) or t not in TARGETS for t in selected):
        raise ValueError('targets must be a nonempty list containing c, phi and/or gamma')
    ordered = list(dict.fromkeys(selected))
    if 'phi' in ordered and 'c' in ordered and ordered.index('phi') > ordered.index('c'):
        ordered.remove('phi');ordered.insert(ordered.index('c'),'phi')
    return tuple(ordered)


def resolve_stages(config, stage='all'):
    options = config.get('stages',{})
    if not isinstance(options,dict) or set(options)-set(STAGE_KEYS.values()):
        raise ValueError('stages supports only 01_collect, 02_prepare, 03_train, 04_distribute')
    if any(type(value) is not bool for value in options.values()):
        raise ValueError('Every stages value must be true or false')
    if stage != 'all':
        if stage not in STAGES: raise ValueError(f'Unknown stage: {stage}')
        return (stage,)
    return tuple(name for name in STAGES if options.get(STAGE_KEYS[name],True))


def run(stage='all', targets=None, config_path=CONFIG/'pipeline.jsonc', download=False, gamma_kind=None):
    from module.output_layout import output_session
    with output_session(RESULTS):
        return _run(stage,targets,config_path,download,gamma_kind)


def _run(stage,targets,config_path,download,gamma_kind):
    from module.model_backend import selected_model, get_backend
    from module.run_artifacts import trained_run, write_json
    from module.output_layout import current_directory
    config = load_jsonc_config(project_path(config_path))
    targets = resolve_targets(config,targets)
    stages = resolve_stages(config,stage)
    if gamma_kind is not None and gamma_kind not in ('wet','dry'):
        raise ValueError('gamma_kind must be wet or dry')
    reuse = config.get('reuse_runs',{})
    if not isinstance(reuse,dict) or set(reuse)-set(TARGETS) or any(v is not None and (not isinstance(v,str) or not v.strip()) for v in reuse.values()):
        raise ValueError('reuse_runs must map c/phi/gamma to null or a saved stage-03 run directory')
    estimator = selected_model()
    backend = get_backend(estimator)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),estimator=estimator,targets=list(targets),
                  enabled_stages=[STAGE_KEYS[s] for s in stages],stages={})
    failed = False; collection_failed = False; preparation = {}
    if 'collect' in stages:
        print('[01_collect]',flush=True)
        try:
            result = collect(config,download)
            if any(item.get('status')=='missing' for item in result.get('files',[])):
                result['status']='incomplete';failed=True;collection_failed=True
        except Exception as error:
            result = dict(status='failed',error=str(error));failed=True;collection_failed=True
        report['stages']['collect'] = result
    if 'prepare' in stages:
        result = {};ready=[]
        for target in targets:
            missing = missing_inputs(target,'prepare')
            if collection_failed:
                result[target] = dict(status='blocked',reason='Collection failed')
            elif missing:
                result[target] = dict(status='missing_input',missing=missing)
            else: ready.append(target)
        if ready:
            print('[02_prepare] '+', '.join(ready),flush=True)
            try: result.update(backend.prepare(ready))
            except Exception as error:
                result.update({t:dict(status='failed',error=str(error)) for t in ready})
                traceback.print_exc()
        preparation = result;report['stages']['prepare']=result
        failed |= any(v['status']!='prepared' for v in result.values())
    current_phi = None
    for target in targets:
        block = ('Collection failed' if collection_failed else
                 'Preparation failed' if target in preparation and preparation[target]['status']!='prepared' else None)
        if target == 'c' and 'phi' in targets and current_phi is None:
            block = 'Current phi/N distribution is required before c; enable 04_distribute for phi. No old-result fallback.'
        run_path = None
        for current in ('train','distribute'):
            if current not in stages: continue
            entries = report['stages'].setdefault(current,{})
            print(f'[{STAGE_KEYS[current]}] {target} ({estimator})',flush=True)
            if block:
                entries[target] = dict(status='blocked',reason=block);failed=True;continue
            try:
                if current == 'train':
                    missing = missing_inputs(target,'prepare')
                    if missing: raise FileNotFoundError(f'Missing training inputs: {missing}')
                    result = backend.train(target,gamma_kind=gamma_kind,n_distribution_run=current_phi)
                    if result['status'] != 'trained': raise RuntimeError('Stage 03 did not complete')
                    run_path = project_path(result['output'])
                else:
                    if run_path is None:
                        run_path = trained_run(target,estimator,reuse.get(target),gamma_kind)
                    result = backend.predict(target,run_path,n_distribution_run=current_phi)
                    if result['status'] != 'complete': raise RuntimeError('Stage 04 did not complete')
                    if target == 'phi': current_phi = result['output']
                entries[target] = result
            except Exception as error:
                entries[target] = dict(status='failed',error=str(error))
                block = f'{STAGE_KEYS[current]} failed';failed=True
                traceback.print_exc()
    report['status'] = 'incomplete' if failed else 'complete'
    directory = current_directory();directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'run_summary.json',report)
    print(f"Status: {report['status']}; report: {directory/'run_summary.json'}",flush=True)
    return 2 if failed else 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all',*STAGES,'compare'],default='all')
    parser.add_argument('--targets',choices=TARGETS,nargs='+')
    parser.add_argument('--gamma-kind',choices=['wet','dry'])
    parser.add_argument('--config',default=str(CONFIG/'pipeline.jsonc'))
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--plan',action='store_true',help='Print selected steps without analysis or result writes')
    parser.add_argument('--comparison-config',default='config/xgboost/evaluation.json')
    parser.add_argument('--models',choices=['xgboost'],nargs='+')
    parser.add_argument('--comparison-targets',choices=['c','N','gamma_wet','gamma_dry'],nargs='+')
    args=parser.parse_args(argv)
    if args.stage=='compare':
        if args.targets or args.gamma_kind or args.download:
            parser.error('Use --comparison-targets with --stage compare')
        from module.model_backend import selected_model
        if selected_model()!='xgboost': parser.error('Independent spatial CV currently supports xgboost only')
        if args.plan:
            print(json.dumps(load_jsonc_config(project_path(args.comparison_config)),ensure_ascii=False,indent=2));return 0
        from module.model_comparison import run_comparison
        return run_comparison(args.comparison_config,args.models,args.comparison_targets)
    if args.models or args.comparison_targets:
        parser.error('--models/--comparison-targets require --stage compare')
    try:
        config=load_jsonc_config(project_path(args.config))
        if not isinstance(config,dict): raise ValueError('Pipeline config must be a JSON object')
        targets=resolve_targets(config,args.targets);stages=resolve_stages(config,args.stage)
        from module.model_backend import selected_model
        estimator=selected_model()
    except (OSError,ValueError) as error: parser.error(str(error))
    if args.plan:
        print(json.dumps(dict(estimator=estimator,stages=[STAGE_KEYS[s] for s in stages],targets=list(targets),
                              reuse_runs=config.get('reuse_runs',{}),analysis_executed=False),ensure_ascii=False,indent=2));return 0
    return run(args.stage,targets,args.config,args.download,gamma_kind=args.gamma_kind)
