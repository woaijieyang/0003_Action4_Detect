"""Counterfactual ONLY: isolate the old off-screen elbow quality condition.

This process never submits a slot, renders formal RGB, or changes a production
predicate.  Results cannot be admitted.  All terminal/table/collision/source
conditions remain active; the two elbow concealment checks are measured rather
than enforced to determine whether asset continuity is the actual blocker.
"""
from pathlib import Path
import sys,os,json,time,collections
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def run(scene):
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import LocalArmVisibility,pose_for_box,projected,junction_outside_frame,upperarm_reenters
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_scene_visibility import SceneVisibility
    from data import action4_coverage1000_sampling as sampling
    from data import action4_coverage1000_domains as domains
    from data.action4_coverage1000_contract import occlusion_ranges,source_eligible
    cfg=json.loads((R/'configs/data/action4_coverage1000_feasibility_wide_v001.json').read_text())
    plan=json.loads((R/'outputs/coverage1000_v001/SLOTS_DRAFT.json').read_text())
    catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text())
    out=R/'outputs/coverage1000_v001/constraint_diagnostic_v001';out.mkdir(exist_ok=True)
    path=out/(scene+'.json');assert not path.exists()
    runtime=Phrase2Runtime(scene);trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']}
    checker=Phrase2CollisionChecker(runtime);visibility=SceneVisibility(runtime,trees,resolution=16)
    def diagnostic_domain(runtime,cfg,side,RB,corners,size,u,width):
        A=np.array(runtime.anchor['sides'][side]['A_brace_from_CAD']);B,C=pose_for_box(RB,A,corners,runtime.K,size,[u,.5],width)
        C[1,3]=0.;B=C@np.linalg.inv(A)
        if not .24<=C[2,3]<=1.4:return [],'optical_Z_range'
        lo,hi=cfg['position_ranges'][side]['v']
        vals=[pose_for_box(RB,A,corners,runtime.K,size,[u,v],width)[1][1,3] for v in [lo,hi]]
        allowed=domains.intersect([tuple(vals)],domains.outside_ring_domain(runtime.arm_terminal_points_B[side],B,runtime.K,size,True))
        allowed=domains.subtract(allowed,domains.table_forbidden(runtime,side,B))
        if not allowed:return [],'terminal_or_table_domain_empty'
        result=[]
        for a,b in allowed:
            values=[]
            for t in [a,b]:
                T=C.copy();T[1,3]=t;_,box,_=projected(corners,T,runtime.K);values.append(float((box[1]+box[3])/(2*size[1])))
            result.append(tuple(values))
        return result,None
    original=sampling.side_guard
    def diagnostic_guard(*args):
        reason=original(*args)
        return None if reason in ['elbow_junction_in_frame','upperarm_reentry'] else reason
    domains.vertical_domains=diagnostic_domain;sampling.side_guard=diagnostic_guard
    selected=[]
    for group in ['C','E']:
        for k in range(4):
            matches=[s for s in plan['slots'] if s['scene_id']==scene and s['group']==group and s['small_width_band']==k and s['high_required']]
            selected.extend(matches[:2])
    rows=[];start=time.monotonic()
    for slot in selected:
        reasons=collections.Counter();success=None
        for attempt in range(128):
            p,error=sampling.propose(runtime,cfg,slot,attempt,catalog,trees,ray_resolution=16)
            if error:reasons[error]+=1;continue
            runtime.apply(p);collision=checker.check()
            if not collision['accepted']:reasons['collision:'+collision['status']]+=1;continue
            sources=visibility.measure(p);reason=None
            for side,m in sources.items():
                lo,hi,last=occlusion_ranges(slot)[side];o=m['estimated_external_occlusion'];counts=m['occluder_class_counts'];n=sum(counts.values())
                if not max(0,lo-.04)<=o<=min(.79,hi+.04):reason=side+':scene_occlusion';break
                if not source_eligible(o,m['primary_source_class']):reason=side+':scene_source';break
                if n>=5 and counts.get('environment',0)+counts.get('other_actor',0)>.5*n:reason=side+':non_own_primary';break
            if reason:reasons[reason]+=1;continue
            size=np.array([runtime.scene.render.resolution_x,runtime.scene.render.resolution_y])
            actual_guards={s:{'junction_outside':bool(junction_outside_frame(runtime,s,np.array(p['actor_pose_capture_by_side'][s]['brace_B_capture']),size)),
                'upperarm_reenters':bool(upperarm_reenters(runtime,s,np.array(p['actor_pose_capture_by_side'][s]['brace_B_capture']),size))} for s in ['left','right']}
            success={'attempt':attempt,'proposal':p,'sources':sources,'elbow_quality_conditions':actual_guards,'NOT_ADMISSIBLE':True}
            break
        rows.append({'slot_id':slot['slot_id'],'group':slot['group'],'small_width_band':slot['small_width_band'],'counterfactual_success':success,'rejections':dict(reasons)})
        path.write_text(json.dumps({'status':'COUNTERFACTUAL_RUNNING_NOT_ADMISSION','scene':scene,'rows':rows,'seconds':time.monotonic()-start}))
        print(scene,slot['slot_id'],bool(success),dict(reasons),flush=True)
    report={'status':'COUNTERFACTUAL_COMPLETE_NOT_ADMISSION','scene':scene,'rows':rows,'seconds':time.monotonic()-start,
        'production_checks_unchanged':True,'images_rendered':0,'interpretation':'Successful counterfactual only isolates an elbow asset/quality dependency. It is not permission to omit the production check.'}
    path.write_text(json.dumps(report));print('COMPLETE',scene,sum(bool(r['counterfactual_success']) for r in rows),len(rows),flush=True)

if __name__=='__main__':run(sys.argv[sys.argv.index('--')+1])
