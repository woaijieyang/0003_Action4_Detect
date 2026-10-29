"""Bounded actual Blender geometry checks; no RGB, no GPU and no publication."""
from pathlib import Path
import sys,json,time,collections,hashlib,os
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
OUT=R/'outputs/phrase2_occlusion500_v001'

def run():
    from data.action4_occlusion500_contract import assert_configuration,occlusion_bin,eligible_sources
    from data.action4_occlusion500_sampling import propose
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import LocalArmVisibility
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_scene_visibility import SceneVisibility
    cfg=json.loads((R/'configs/data/action4_phrase2_occlusion500_v001.json').read_text());assert_configuration(cfg)
    plan=json.loads((OUT/'SLOTS.json').read_text());catalog=json.loads((R/os.environ.get('OC500_CATALOG_OVERRIDE',cfg['occlusion_catalog'])).read_text())
    begin=time.time();results=[];failures=collections.Counter()
    for scene in cfg['scene_counts']:
        runtime=Phrase2Runtime(scene);trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']};checker=Phrase2CollisionChecker(runtime);visibility=SceneVisibility(runtime,trees)
        # One slot for every left bin in each scene. Right classes and all
        # continuous proposals remain the real frozen random sampler's output.
        selected=[next(r for r in plan['slots'] if r['scene_id']==scene and r['occlusion_bins']['left']==i) for i in range(5)]
        for slot in selected:
            row={'slot_id':slot['slot_id'],'scene':scene,'requested_bins':slot['occlusion_bins'],'attempts':[]};accepted=[]
            for attempt in range(8):
                proposal,error=propose(runtime,cfg,slot,attempt,catalog,local_visibility=trees)
                if error:failures[error]+=1;row['attempts'].append({'attempt':attempt,'ordinary_rejection':error});continue
                runtime.apply(proposal);collision=checker.check()
                if not collision['accepted']:failures[collision['status']]+=1;row['attempts'].append({'attempt':attempt,'ordinary_rejection':collision['status']});continue
                sources=visibility.measure(proposal);ok,reason=eligible_sources(sources,slot['occlusion_bins'])
                if not ok:failures[reason]+=1;row['attempts'].append({'attempt':attempt,'ordinary_rejection':reason});continue
                corners=runtime.export_corners();assert corners['max_proposal_vs_actual_error_px']<=.005
                for side in ['left','right']:
                    B=np.array(proposal['actor_pose_capture_by_side'][side]['brace_B_capture']);A=np.array(proposal['sides'][side]['A_brace_from_CAD']);C=np.array(proposal['actor_pose_capture_by_side'][side]['core_C_capture'])
                    assert np.allclose(B@A,C,atol=1e-9,rtol=0)
                accepted.append(proposal)
                row['attempts'].append({'attempt':attempt,'geometry_pass':True,'actual_projection_error_px':corners['max_proposal_vs_actual_error_px'],
                    'by_side':proposal['phrase2']['by_side'],'core_capture':{s:proposal['actor_pose_capture_by_side'][s]['core_C_capture'] for s in ['left','right']},'sources':sources})
                if len(accepted)==2:break
            if len(accepted)==2:
                a,b=accepted
                changes={s:{k:a['phrase2']['by_side'][s][k]!=b['phrase2']['by_side'][s][k] for k in ['bbox_center_uv','actual_width_fraction','forearm_roll_deg','forearm_direction_capture']} for s in ['left','right']}
                assert all(all(v.values()) for v in changes.values()),'Only occlusion changed while other continuous parameters stayed fixed'
                row['two_valid_attempts_change_position_size_orientation_both_sides']=changes
            row['geometry_pass_count']=len(accepted);results.append(row)
            temp=OUT/'GEOMETRY_RANDOMNESS_PREFLIGHT.json.tmp';temp.write_text(json.dumps({'status':'RUNNING_GEOMETRY_ONLY','rows':results,'failures':dict(failures),'seconds':time.time()-begin}));os.replace(temp,OUT/'GEOMETRY_RANDOMNESS_PREFLIGHT.json')
            print(scene,slot['slot_id'],slot['occlusion_bins'],len(accepted),dict(failures),flush=True)
    passed=sum(r['geometry_pass_count'] for r in results)
    report={'status':'GEOMETRY_PREFLIGHT_FINISHED_REVIEW_REQUIRED','rows':results,'valid_proposals':passed,'failures':dict(failures),'seconds':time.time()-begin,
            'actual_RGB_or_mask_rendered':False,'not_proven':'Actual mask occlusion bins, image quality and formal acceptance require the scheduled render preview.'}
    (OUT/'GEOMETRY_RANDOMNESS_PREFLIGHT.json').write_text(json.dumps(report,indent=2));print('PREFLIGHT_FINISHED',passed)

if __name__=='__main__':run()
