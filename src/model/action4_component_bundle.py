"""Bind independently selected components without merging checkpoint weights."""
from pathlib import Path
import hashlib,json
import torch
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.action4_components import component_state


def verify_feature_parent(a0,legacy):
    if a0.get('role')!='A0' or a0.get('schema_version')!='action4.component_weights.v1':
        raise ValueError('Geometry needs an exported A0 component')
    expected=component_state(legacy,'A0');actual=a0['model']
    if set(expected)!=set(actual) or any(not torch.equal(v,actual[k]) for k,v in expected.items()):
        raise ValueError('Geometry feature parent parameters differ, even if source labels match')


def file_identity(path):
    path=contained_path(R,Path(path),exists=True);digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return {'path':str(path.relative_to(R)),'sha256':digest.hexdigest(),'bytes':path.stat().st_size}


def create_bundle(directory,destination):
    directory=contained_path(R,Path(directory),exists=True)
    destination=contained_path(R,Path(destination),exists=False)
    if destination.exists():raise FileExistsError(destination)
    states={n:torch.load(directory/(n+'.pt'),map_location='cpu',weights_only=True) for n in ['A0','B0','residual']}
    a,b,g=states['A0'],states['B0'],states['residual']
    if b.get('role')!='B0' or b.get('schema_version')!='action4.component_eval.v1':raise ValueError('Expected B0 eval weights')
    if g.get('schema')!='action4.residual_predicted_roi.v1':raise ValueError('Unknown geometry schema')
    if a['provenance']['source']!=g['source_checkpoint']:raise ValueError('Geometry parent reference mismatch')
    parent=contained_path(R,g['source_checkpoint'],exists=True)
    verify_feature_parent(a,torch.load(parent,map_location='cpu',weights_only=True))
    if a['preprocess']['roi_size']!=448 or a['preprocess']['roi_expansion']!=2.25:raise ValueError('Geometry preprocessing mismatch')
    bundle={'schema':'action4.independent_bundle.v1','components':{n:file_identity(directory/(n+'.pt')) for n in states},'geometry_feature_parent':file_identity(parent),'parameter_match_verified':True,'B0_step':b['step'],'geometry_step':g['step'],'A0_preprocess':a['preprocess'],'B0_preprocess':b['config']['preprocess'],'B0_decode':b['config']['decode'],'merged_checkpoint':False,'production_accepted':False,'scope':'fixed diagnostic bundle; known real failures retained'}
    with destination.open('x') as handle:json.dump(bundle,handle,indent=2)
    return bundle
