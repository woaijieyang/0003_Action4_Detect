"""Immutable code/config consumption for the existing Action4 trainer and workers."""
import hashlib,json,os,shutil,subprocess,sys
from functools import lru_cache
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R, CODE_ROOT, contained_path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def child_pythonpath():
    return str(CODE_ROOT/'runtime/surface_deps')+os.pathsep+str(CODE_ROOT/'src')

@lru_cache(maxsize=8)
def _verify(spec_path):
    spec=json.loads(contained_path(R,spec_path,exists=True).read_text())
    root=contained_path(R,spec['code_root'],exists=True)
    if root!=CODE_ROOT:raise ValueError('Run imported mutable/non-selected code root')
    if spec['project_root']!=str(R):raise ValueError('Snapshot data-root mismatch')
    for name,sha in spec['files'].items():
        path=contained_path(R,root/name,exists=True)
        if digest(path)!=sha:raise ValueError('Immutable run content changed: '+name)
    return spec

def verify_running_snapshot(cfg):
    if not cfg.get('run_spec'):
        if cfg.get('require_immutable_source'):raise ValueError('Missing immutable RUN_SPEC')
        return
    spec=_verify(cfg['run_spec'])
    config=contained_path(R,spec['config'],exists=True)
    if json.loads(config.read_text())!=cfg:raise ValueError('Consumed config differs from immutable resolved configuration')
    for module in list(sys.modules.values()):
        name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
        if path and name.split('.')[0] in ('data','model','eval'):
            if not Path(path).resolve().is_relative_to(CODE_ROOT):raise ValueError('Mixed project module origins: '+name)

def corner_eval_source_manifest(root,run_spec_path):
    """Code-only, no resolved config/RunSpec hash cycle; used only for corner opt-in."""
    root=contained_path(R,root,exists=True)
    if not (root/'src/eval/action4_image_guided_eval.py').is_file():
        raise ValueError('Tested corner source lacks its evaluation orchestrator')
    files={}
    for folder in ('src','scripts','runtime/surface_deps'):
        base=root/folder
        if not base.exists():continue
        for p in sorted(base.rglob('*.py')):
            if p.is_symlink():raise ValueError('Corner source cannot contain symlink code')
            files[str(p.relative_to(root))]=digest(p)
    value={'schema':'action4.corner_inference_source.v1','code_root':str(root.relative_to(R)),
           'run_spec':str(Path(run_spec_path).relative_to(R)),'files_sha256':files}
    value['identity']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    path=root/'CORNER_EVAL_SOURCE.json'
    if path.exists():raise FileExistsError('Preserve existing corner source manifest')
    path.write_text(json.dumps(value,indent=2))
    return {'path':str(path.relative_to(R)),'sha256':digest(path)}

def freeze(config,run_id,source_spec=None):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Unsafe run ID')
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    corner=cfg.get('a0_variant')=='a0_corner_image' or cfg.get('corner_enabled') is True
    if corner and (cfg.get('role')!='A0' or cfg.get('sam_enabled') is not True or cfg.get('corner_enabled') is not True):
        raise ValueError('Corner freeze requires consistent A0/SAM/corner flags')
    base=contained_path(R,Path('runtime/action4_runs')/run_id,exists=False)
    base.mkdir(parents=True,exist_ok=False);root=base/'source';root.mkdir()
    source_root=R
    if source_spec:
        previous=json.loads(contained_path(R,source_spec,exists=True).read_text())
        source_root=contained_path(R,previous['code_root'],exists=True)
        for name,sha in previous['files'].items():
            if digest(contained_path(R,source_root/name,exists=True))!=sha:raise ValueError('Tested source snapshot mutated')
    # Reuse an actually tested immutable source when requested; never silently use a newer worktree.
    folders=[('src',{'.py'}),('configs',{'.json','.yaml','.yml'}),('runtime/surface_deps',None)]
    if corner:folders.append(('scripts',{'.py'}))
    for folder,suffixes in folders:
        source=source_root/folder
        for current,dirs,files in os.walk(source,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d!='__pycache__' and not (Path(current)/d).is_symlink())
            for name in sorted(files):
                p=Path(current)/name
                if p.is_symlink() or (suffixes is not None and p.suffix not in suffixes):continue
                rel=p.relative_to(source_root);target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,target)
    inputs=root/'inputs';inputs.mkdir()
    registry=json.loads(contained_path(R,cfg.get('parent_registry','configs/model/action4_best_registry_v003.json'),exists=True).read_text())
    role=cfg['role'];entry=registry['entries'][role]
    original=contained_path(R,cfg['parent_bundle'],exists=True)
    if digest(original)!=entry['sha256'] or cfg['parent_best_step']!=entry['selected_step']:raise ValueError('Selected parent identity mismatch')
    if corner:
        previous_sources=cfg.get('corner_eval_freeze_sources')
        cfg['corner_eval_freeze_sources']={'parent':{'path':str(original.relative_to(R)),'sha256':entry['sha256'],'selected_step':cfg['parent_best_step']}}
        if previous_sources is not None:cfg['corner_eval_freeze_sources']['previous']=previous_sources
    if corner and cfg.get('parent_bundle_sha256')!=entry['sha256']:
        raise ValueError('Corner parent SHA differs from selected actual parent')
    parent_aliases=[]
    if corner:
        for key,ref in cfg.items():
            if isinstance(ref,dict) and {'path','sha256'}<=set(ref) and 'parent' in key.lower():
                if contained_path(R,ref['path'],exists=False)==original:
                    if ref['sha256']!=entry['sha256'] or ('selected_step' in ref and ref['selected_step']!=cfg['parent_best_step']):
                        raise ValueError('Conflicting duplicate corner parent binding')
                    parent_aliases.append(key)
    target=inputs/(role+'_parent.pt');shutil.copyfile(original,target)
    cfg['parent_bundle']=str(target.relative_to(R));entry=dict(entry,path=cfg['parent_bundle'])
    for key in parent_aliases:cfg[key]=dict(cfg[key],path=cfg['parent_bundle'])
    parent_registry={'schema':'action4.selected_parent_registry.v1','entries':{role:entry}}
    rp=inputs/'parent_registry.json';rp.write_text(json.dumps(parent_registry))
    cfg['parent_registry']=str(rp.relative_to(R))
    if cfg.get('evaluation_counterpart_bundle'):
        p=contained_path(R,cfg['evaluation_counterpart_bundle'],exists=True)
        if digest(p)!=cfg['evaluation_counterpart_sha256']:raise ValueError('Counterpart identity mismatch')
        if corner:
            ref=cfg.get('corner_counterpart',{}).get('B0')
            if not isinstance(ref,dict) or contained_path(R,ref['path'],exists=True)!=p or ref['sha256']!=cfg['evaluation_counterpart_sha256']:
                raise ValueError('Corner counterpart differs before freeze copy')
        if corner:cfg['corner_eval_freeze_sources']['B0']={'path':str(p.relative_to(R)),'sha256':cfg['evaluation_counterpart_sha256']}
        t=inputs/'B0_counterpart.pt';shutil.copyfile(p,t);cfg['evaluation_counterpart_bundle']=str(t.relative_to(R))
        if corner:cfg['corner_counterpart']=dict(cfg['corner_counterpart'],B0={'path':cfg['evaluation_counterpart_bundle'],'sha256':cfg['evaluation_counterpart_sha256']})
    cfg['source_identity']={str(p.relative_to(R)):digest(p) for p in sorted((root/'src').rglob('*.py'))}
    code_hashes={str(p.relative_to(root)):digest(p) for p in sorted((root/'src').rglob('*.py'))}
    code_id=hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest()
    if cfg.get('tested_project_code_sha256') and cfg['tested_project_code_sha256']!=code_id:
        raise ValueError('Formal source differs from actually tested source')
    cfg['project_code_sha256']=code_id
    cfg['validation_receipt_sha256']=digest(contained_path(R,cfg['validation_receipt'],exists=True))
    cfg['require_immutable_source']=True;cfg['run_spec']=str((base/'RUN_SPEC.json').relative_to(R))
    if corner:cfg['corner_eval_source']=corner_eval_source_manifest(root,base/'RUN_SPEC.json')
    cp=root/'resolved_config.json';cp.write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
    files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    spec={'schema':'action4.immutable_run.v1','run_id':run_id,'project_root':str(R),
          'code_root':str(root.relative_to(R)),'config':str(cp.relative_to(R)),
          'files':files,'project_code_sha256':code_id,'git_head_context_only':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
          'environment':{'python':sys.version},'mutable_worktree_not_used_by_children':True}
    (base/'RUN_SPEC.json').write_text(json.dumps(spec,indent=2))
    launcher=base/'launch.py'
    launcher.write_text('''import os,json,sys,runpy
from pathlib import Path
root=Path('/data/zy/vln/a-test').resolve(strict=True)
spec_path=Path(__file__).resolve().with_name('RUN_SPEC.json')
if not spec_path.is_relative_to(root):raise ValueError('Run outside authorized root')
spec=json.loads(spec_path.read_text());code=(root/spec['code_root']).resolve(strict=True)
os.environ['ACTION4_PROJECT_ROOT']=str(root)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
sys.path[:]=[str(code/'runtime/surface_deps'),str(code/'src')]+[p for p in sys.path if p and not Path(p).resolve().is_relative_to(root/'src') and Path(p).resolve()!=root]
os.environ['PYTHONPATH']=str(code/'runtime/surface_deps')+':'+str(code/'src')
from model.training.action4_run_snapshot import verify_running_snapshot
cfg=json.loads((root/spec['config']).read_text());verify_running_snapshot(cfg)
entry=sys.argv[1] if len(sys.argv)>1 else 'model.training.action4_stage2_dispatch'
if entry not in ('model.training.action4_stage2_dispatch','model.training.action4_stage2_train','eval.action4_stage2_eval','eval.action4_image_guided_eval','eval.action4_corner_rescore','eval.action4_corner_capacity','eval.action4_joint_capacity_probe','eval.action4_staged_video','eval.action4_staged_video_parallel'):raise ValueError('Unapproved run entry')
sys.argv=[entry,'--selection' if entry in ('eval.action4_staged_video','eval.action4_staged_video_parallel') else '--config',spec['config']]+sys.argv[2:]
runpy.run_module(entry,run_name='__main__')
''')
    for p in root.rglob('*'):
        if p.is_file():p.chmod(0o444)
    (base/'RUN_SPEC.json').chmod(0o444);launcher.chmod(0o444)
    return {'run_spec':str((base/'RUN_SPEC.json').relative_to(R)),'launcher':str(launcher.relative_to(R)),
            'config':str(cp.relative_to(R)),'source_files':len(files)}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args();print(json.dumps(freeze(a.config,a.run_id)))
