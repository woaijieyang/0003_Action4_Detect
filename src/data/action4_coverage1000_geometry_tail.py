"""Continue only unfinished geometric attempts while the RGB previews are reviewed.

No RGB, no queue ownership, no formal labels or publication. Frozen sampler,
production-resolution local rays, actual scene/collision and corner checks.
"""
from pathlib import Path
import argparse,collections,json,os,sys,time
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def run(scene,shard,shards):
    from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
    from data.action4_coverage1000_sampling import propose
    from data.action4_coverage1000_contract import occlusion_ranges,source_eligible
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_scene_visibility import SceneVisibility
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());out=R/'outputs/coverage1000_v001/geometry_tail_v001';out.mkdir(exist_ok=True)
    plan=json.loads((R/cfg['plan_path']).read_text());catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text())
    selected=[s for s in plan['slots'] if s['scene_id']==scene and plan['initial_attempt_by_slot'][str(s['slot_id'])]==64][shard::shards]
    path=out/f'{scene}_{shard:02d}.json';assert not path.exists();start=time.monotonic()
    runtime=CoverageArmRuntime(scene,require_visual_admission=True,asset_version=cfg['arm_asset_version'])
    trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']};checker=Phrase2CollisionChecker(runtime);visibility=SceneVisibility(runtime,trees)
    rows=[];failures=collections.Counter()
    for slot in selected:
        reasons=collections.Counter();success=None;last=64
        for attempt in range(64,576):
            last=attempt+1;p,error=propose(runtime,cfg,slot,attempt,catalog,trees,ray_resolution=28)
            if error:reasons[error]+=1;continue
            runtime.apply(p);collision=checker.check()
            if not collision['accepted']:reasons['collision:'+collision['status']]+=1;continue
            sources=visibility.measure(p);reason=None
            for side,m in sources.items():
                lo,hi,last_inclusive=occlusion_ranges(slot)[side];o=m['estimated_external_occlusion'];counts=m['occluder_class_counts'];n=sum(counts.values())
                if not max(0,lo-.04)<=o<=min(.79,hi+.04):reason=side+':scene_occlusion';break
                if not source_eligible(o,m['primary_source_class']):reason=side+':scene_source';break
                if o>.35 and m['primary_occluder_object']==side+'_forearm_continuation':reason=side+':upperarm_not_forearm_occlusion';break
                if n>=5 and counts.get('environment',0)+counts.get('other_actor',0)>.5*n:reason=side+':non_own_primary';break
            if reason:reasons[reason]+=1;continue
            actual=runtime.export_corners()
            if actual['max_proposal_vs_actual_error_px']>.005:raise ValueError('Actual export differs from frozen proposal')
            success={'attempt':attempt,'pair_metrics':p['phrase2']['pair_metrics'],'actual_projection_error_px':actual['max_proposal_vs_actual_error_px'],
                'C_by_side':{s:p['actor_pose_capture_by_side'][s]['core_C_capture'] for s in ['left','right']},'sources':sources};break
        failures.update(reasons);rows.append({'slot_id':slot['slot_id'],'group':slot['group'],'success':success,'next_untried_attempt':last,'rejections':dict(reasons)})
        temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps({'status':'GEOMETRY_TAIL_RUNNING','protocol_id':cfg['protocol_id'],'rows':rows,'failures':dict(failures),'seconds':time.monotonic()-start}));os.replace(temp,path)
    report={'status':'GEOMETRY_TAIL_COMPLETE_NOT_IMAGE_ACCEPTANCE','protocol_id':cfg['protocol_id'],'rows':rows,'failures':dict(failures),'seconds':time.monotonic()-start,
        'successful_slots':sum(bool(r['success']) for r in rows),'no_RGB_rendered':True,'queue_not_modified':True}
    path.write_text(json.dumps(report));print(scene,shard,len(rows),report['successful_slots'],report['seconds'],flush=True)

if __name__=='__main__':
    a=sys.argv[sys.argv.index('--')+1:];run(a[0],int(a[1]),int(a[2]))
