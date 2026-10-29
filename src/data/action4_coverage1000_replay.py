"""Single-frame frozen-seed replay. Geometry-only by default; never overwrites data."""
from pathlib import Path
import argparse,hashlib,json,os,sys
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def replay(sample_id,manifest_role,render):
    if Path.cwd().resolve()!=R:raise ValueError('Project root required')
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());plan=json.loads((R/cfg['plan_path']).read_text())
    slot=next(s for s in plan['slots'] if s['sample_id']==sample_id)
    if manifest_role=='formal':
        if json.loads((R/'data/phrase2/DATASET_COMPLETE.json').read_text()).get('images')!=6500:raise ValueError('Formal6500 not published; explicitly use --manifest-role candidate')
        manifest=R/'data/phrase2'/slot['split']/'annotations.jsonl'
    else:manifest=R/cfg['candidate_root']/slot['split']/'annotations.jsonl'
    record=next(r for line in manifest.open() if (r:=json.loads(line))['sample_id']==sample_id)
    if record.get('batch_id')!=cfg['batch_id']:raise ValueError('Replay only this new batch, never old5500 with repaired assets')
    if render:
        if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):raise ValueError('Explicit authorized idle GPU UUID required for optional diagnostic rendering')
        from data.action4_phrase2_acceleration import install
        install(__file__)
    from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
    from data.action4_coverage1000_sampling import propose
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    runtime=CoverageArmRuntime(slot['scene_id'],True,cfg['arm_asset_version']);checker=Phrase2CollisionChecker(runtime)
    catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text());p,error=propose(runtime,cfg,slot,record['attempt'],catalog)
    if error:raise ValueError('Frozen accepted parameters failed to reproduce: '+error)
    runtime.apply(p);collision=checker.check()
    if not collision['accepted']:raise ValueError('Frozen accepted collision geometry differs')
    actual=runtime.export_corners();maximum=0.
    for obj in record['instances']:
        T=np.array(actual['canonical8_by_instance'][str(obj['instance_id'])]['object_to_capture_cv_m']);X=np.array(p['canonical8_m'])
        cam=X@T[:3,:3].T+T[:3,3];uv=cam@runtime.K.T;uv=uv[:,:2]/uv[:,2:]
        maximum=max(maximum,float(np.max(abs(uv-np.array(obj['corners_px'])))))
    if maximum>.005:raise ValueError('Independent replay differs from recorded actual GT')
    out=R/'outputs/coverage1000_v001/replays'/f"{sample_id}_a{record['attempt']:05d}";out.mkdir(parents=True,exist_ok=False)
    report={'status':'PASS_FROZEN_GEOMETRY_AND_SEED_REPLAY','sample_id':sample_id,'attempt':record['attempt'],'render_seed':p['entry']['seed'],
        'protocol_id':cfg['protocol_id'],'max_saved_vs_replayed_corner_error_px':maximum,'collision':collision['status'],'RGB_rerendered':False,
        'definition':'Geometry replay proves the frozen parameter/camera/installation/corner chain, not unexecuted RGB bit identity or model accuracy.'}
    if render:
        from data import action4_phrase2_pixels as pixels
        entry=dict(p['entry'],sample_id='replay_'+sample_id);(out/'pixels').mkdir();(out/'runtime').mkdir();canonical=runtime.metadata['canonical_definition']
        ids={path:hashlib.sha256((R/path).read_bytes()).hexdigest() for path in pixels.REQUIRED_SOURCE_PATHS};ids[canonical['source_manifest']['path']]=canonical['source_manifest']['sha256']
        ids[runtime.coverage_overlay_ref['path']]=runtime.coverage_overlay_ref['sha256'];ids[runtime.metadata_path]=hashlib.sha256((R/runtime.metadata_path).read_bytes()).hexdigest()
        result=pixels.render_frame(scene=runtime.scene,entry=entry,targets=runtime.targets(),canonical=canonical,K=runtime.K.tolist(),data_out=out/'pixels',runtime_out=out/'runtime',output_role='diagnostic',source_identities=ids,
            render_settings={'device':'CUDA','denoiser':'OPTIX','samples':32,'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16,'seed':entry['seed'],'worker_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']},verify_scene=runtime.verify)
        report.update(RGB_rerendered=True,diagnostic_image=result['record']['image'],formal_dataset_modified=False)
    (out/'REPLAY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:];parser=argparse.ArgumentParser();parser.add_argument('--sample-id',required=True);parser.add_argument('--manifest-role',choices=['formal','candidate'],default='formal');parser.add_argument('--render',action='store_true');a=parser.parse_args(args);replay(a.sample_id,a.manifest_role,a.render)
