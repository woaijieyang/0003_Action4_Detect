"""Bounded actual-scene feasibility probe. No GPU, RGB, publication or training."""
from pathlib import Path
import os,sys,json,time,collections,argparse
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def run(scene,shard,shards,attempts,output,plan_path,config_path):
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_scene_visibility import SceneVisibility
    from data.action4_coverage1000_sampling import propose
    from data.action4_coverage1000_contract import occlusion_ranges,source_eligible
    cfg=json.loads((R/config_path).read_text())
    if cfg.get('arm_runtime')=='coverage1000_continuous_upperarm_v001':
        from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
        Phrase2Runtime=CoverageArmRuntime
    plan=json.loads((R/plan_path).read_text())
    catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text())
    out=(R/output).resolve();assert out.is_relative_to(R/'outputs/coverage1000_v001');out.mkdir(exist_ok=True)
    destination=out/f'{scene}_{shard:02d}.json'
    if destination.exists():raise FileExistsError(destination)
    start=time.monotonic();runtime=Phrase2Runtime(scene)
    trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']}
    checker=Phrase2CollisionChecker(runtime);visibility=SceneVisibility(runtime,trees,resolution=16)
    # Every group, size/ratio/high/gap category appears in the complete bounded scan.
    selected=[s for s in plan['slots'] if s['scene_id']==scene]
    selected=selected[shard::shards]
    rows=[];failures=collections.Counter();accepted=[]
    for index,slot in enumerate(selected):
        row={'slot_id':slot['slot_id'],'group':slot['group'],'small_width_band':slot.get('small_width_band'),
             'ratio_band':slot.get('ratio_band'),'gap_band':slot.get('gap_band'),'high_band':slot.get('high_band'),
             'high_required':slot['high_required'],'designated_side':slot['designated_side'],'accepted':False,'rejections':{}}
        reasons=collections.Counter()
        for attempt in range(attempts):
            proposal,error=propose(runtime,cfg,slot,attempt,catalog,local_visibility=trees,ray_resolution=16)
            if error:reasons[error]+=1;continue
            runtime.apply(proposal);collision=checker.check()
            if not collision['accepted']:reasons['collision:'+collision['status']]+=1;continue
            sources=visibility.measure(proposal);reason=None
            for side,m in sources.items():
                lo,hi,last=occlusion_ranges(slot)[side];o=m['estimated_external_occlusion']
                if not max(0,lo-.04)<=o<=min(.79,hi+.04):reason=side+':scene_ray_occlusion';break
                if not source_eligible(o,m['primary_source_class']):reason=side+':scene_ray_source';break
                counts=m['occluder_class_counts'];occluded=sum(counts.values())
                if occluded>=5 and counts.get('environment',0)+counts.get('other_actor',0)>.5*occluded:reason=side+':non_own_primary';break
            if reason:reasons[reason]+=1;continue
            actual=runtime.export_corners()
            if actual['max_proposal_vs_actual_error_px']>.005:raise ValueError('Actual scene projection mismatch')
            row.update(accepted=True,attempt=attempt,pair=proposal['phrase2']['pair_metrics'],notes=proposal['phrase2']['by_side'],sources=sources,
                       projection_error_px=actual['max_proposal_vs_actual_error_px'])
            accepted.append({'slot_id':slot['slot_id'],'attempt':attempt,'proposal':proposal})
            break
        row['rejections']=dict(reasons);failures.update(reasons);rows.append(row)
        if index%10==0:print(scene,shard,index,len(accepted),dict(failures),flush=True)
        report={'status':'GEOMETRY_ONLY_RUNNING','scene':scene,'shard':shard,'pid':os.getpid(),'rows':rows,'accepted':len(accepted),
                'failures':dict(failures),'seconds':time.monotonic()-start,'formal_RGB_rendered':False}
        temp=destination.with_suffix('.json.tmp');temp.write_text(json.dumps(report));os.replace(temp,destination)
    report.update(status='BOUNDED_GEOMETRY_SCAN_COMPLETE_NOT_PIXEL_ACCEPTANCE',seconds=time.monotonic()-start)
    destination.write_text(json.dumps(report));destination.with_name(destination.stem+'_proposals.json').write_text(json.dumps(accepted))
    print('COMPLETE',scene,shard,len(accepted),len(rows),report['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4);p.add_argument('--attempts',type=int,default=16)
    p.add_argument('--output',default='outputs/coverage1000_v001/feasibility_v004');p.add_argument('--plan',default='outputs/coverage1000_v001/SLOTS_DRAFT.json')
    p.add_argument('--config',default='configs/data/action4_coverage1000_v001.json')
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    a=p.parse_args(args);run(a.scene,a.shard,a.shards,a.attempts,a.output,a.plan,a.config)
