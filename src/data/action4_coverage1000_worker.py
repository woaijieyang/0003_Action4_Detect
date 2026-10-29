"""Persistent fixed-template renderer for frozen coverage slots only."""
from pathlib import Path
import os,sys,json,time,hashlib,collections,traceback
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))


def run(scene,worker_id):
    from data.action4_phrase2_acceleration import install
    install(__file__)
    from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_phrase2_scene_visibility import SceneVisibility
    from data.action4_phrase2_surface import SurfaceDeriver
    from data import action4_phrase2_pixels as pixels
    from data.action4_coverage1000_sampling import propose
    from data.action4_coverage1000_contract import occlusion_ranges,source_eligible,actual_accept
    from data import action4_coverage1000_queue as queue
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());out=R/cfg['output_root']
    if cfg['status']!='FROZEN_FOR_PREVIEW' or not cfg['formal_rgb_allowed']:raise ValueError('Geometry feasibility and frozen protocol required')
    catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text());runtime=CoverageArmRuntime(scene,require_visual_admission=True,asset_version=cfg['arm_asset_version'])
    if runtime.metadata.get('phrase2_arm_geometry_version')!='coverage1000_continuous_upperarm_v001':raise ValueError('Independently admitted continuous arm asset required')
    checker=Phrase2CollisionChecker(runtime);trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']}
    source_measure=SceneVisibility(runtime,trees);deriver=SurfaceDeriver();canonical=runtime.metadata['canonical_definition']
    ids={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in pixels.REQUIRED_SOURCE_PATHS};ids[canonical['source_manifest']['path']]=canonical['source_manifest']['sha256']
    ids[runtime.coverage_overlay_ref['path']]=runtime.coverage_overlay_ref['sha256'];ids[runtime.metadata_path]=hashlib.sha256((R/runtime.metadata_path).read_bytes()).hexdigest()
    source_paths=['src/data/action4_coverage1000_worker.py','src/data/action4_coverage1000_sampling.py','src/data/action4_coverage1000_domains.py','src/data/action4_coverage1000_contract.py','src/data/action4_coverage1000_queue.py','configs/data/action4_coverage1000_v001.json',runtime.metadata_path]
    sources={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in source_paths}
    directory=out/'workers'/worker_id;directory.mkdir(parents=True,exist_ok=False)
    (directory/'INITIALIZED.json').write_text(json.dumps({'pid':os.getpid(),'scene':scene,'GPU':os.environ['CUDA_VISIBLE_DEVICES'],'sources':sources,'asset':runtime.metadata['blend_ref'],'one_scene_load':True,'cpu_affinity':sorted(os.sched_getaffinity(0))},indent=2))
    counters=collections.Counter();rendered=0;begin=time.monotonic()
    while not(out/'STOP.json').exists():
        claim=queue.claim(scene)
        if claim is None:
            if not queue.pending_enabled(scene):break
            time.sleep(1);continue
        slot=json.loads(claim['spec']);submitted=False
        try:
            for _ in range(cfg['max_attempts_per_claim']):
                attempt=queue.next_attempt(slot['slot_id'])
                if attempt>=cfg['max_total_attempts_per_slot']:
                    queue.transition(slot['slot_id'],'unfilled',error='bounded_attempts_exhausted');submitted=True;break
                proposal,error=propose(runtime,cfg,slot,attempt,catalog,local_visibility=trees,ray_resolution=28)
                if error:counters[error]+=1;continue
                runtime.apply(proposal);collision=checker.check()
                if not collision['accepted']:counters['collision:'+collision['status']]+=1;continue
                sources_measured=source_measure.measure(proposal);reason=None
                for side,m in sources_measured.items():
                    lo,hi,last=occlusion_ranges(slot)[side];o=m['estimated_external_occlusion'];counts=m['occluder_class_counts'];n=sum(counts.values())
                    if not max(0,lo-.04)<=o<=min(.79,hi+.04):reason=side+':scene_occlusion';break
                    if not source_eligible(o,m['primary_source_class']):reason=side+':scene_source';break
                    if o>.35 and m['primary_occluder_object']==side+'_forearm_continuation':reason=side+':upperarm_not_forearm_occlusion';break
                    if n>=5 and counts.get('environment',0)+counts.get('other_actor',0)>.5*n:reason=side+':non_own_primary';break
                if reason:counters[reason]+=1;continue
                actual=runtime.export_corners()
                if actual['max_proposal_vs_actual_error_px']>.005:raise ValueError('Scene projection identity failed')
                case=directory/f"{slot['sample_id']}_a{attempt:05d}";case.mkdir()
                staging=R/cfg['candidate_root']/slot['split']/'.staging'/slot['sample_id']/f'a{attempt:05d}';staging.mkdir(parents=True,exist_ok=False)
                (case/'FRAME_SNAPSHOT.json').write_text(json.dumps({'slot':slot,'parameters':proposal,'collision':collision,'actual_corner_export':actual,'sources':sources},indent=2))
                settings={'device':'CUDA','denoiser':'OPTIX','samples':cfg['rgb_samples'],'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16,'seed':proposal['entry']['seed'],'worker_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
                start=time.monotonic();result=pixels.render_frame(scene=runtime.scene,entry=proposal['entry'],targets=runtime.targets(),canonical=canonical,K=runtime.K.tolist(),data_out=staging,runtime_out=case,output_role='dataset_candidate',source_identities=ids,render_settings=settings,verify_scene=runtime.verify)
                record=result['record'];record['occlusion_sources']=sources_measured
                for obj in record['instances']:
                    side=['left','right'][obj['instance_id']-1]
                    m=sources_measured[side]
                    if m['sampled_amodal_body_rays']>50 and abs(obj['occlusion_inframe']-m['estimated_external_occlusion'])>.15:raise ValueError('Actual-mask/scene-ray engineering mismatch')
                good,_,_=actual_accept(slot,record)
                if good:record=deriver.derive(record,staging/'geometry')
                record.update(frame_snapshot_ref=str((case/'FRAME_SNAPSHOT.json').relative_to(R)),asset_version='coverage1000_continuous_upperarm_v001_light_v001',
                    coverage_asset_metadata_ref=runtime.metadata_path,coverage_arm_overlay_ref=runtime.coverage_overlay_ref,coverage_protocol_id=cfg['protocol_id'])
                payload={'slot':slot,'record':record,'proposal':proposal,'collision':collision,'actual_projection_error_px':actual['max_proposal_vs_actual_error_px'],
                    'runtime_dir':str(case.relative_to(R)),'staging_dir':str(staging.relative_to(R)),'render_seconds':time.monotonic()-start}
                pointer=case/'CANDIDATE.json';pointer.write_text(json.dumps(payload,allow_nan=False))
                queue.transition(slot['slot_id'],'rendered',record={'candidate':str(pointer.relative_to(R))});rendered+=1;submitted=True;break
            if not submitted:queue.transition(slot['slot_id'],'pending')
        except BaseException:
            failure={'pid':os.getpid(),'slot':slot,'traceback':traceback.format_exc()};(directory/'FAILURE.json').write_text(json.dumps(failure,indent=2));(out/'STOP.json').write_text(json.dumps(failure,indent=2))
            queue.transition(slot['slot_id'],'failed',error=failure['traceback']);raise
        temp=directory/'STATUS.json.tmp';temp.write_text(json.dumps({'pid':os.getpid(),'rendered':rendered,'rejections':dict(counters),'elapsed_seconds':time.monotonic()-begin}));os.replace(temp,directory/'STATUS.json')
    (directory/'COMPLETE.json').write_text(json.dumps({'rendered':rendered,'rejections':dict(counters),'seconds':time.monotonic()-begin}))

if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:];run(args[0],args[1])
