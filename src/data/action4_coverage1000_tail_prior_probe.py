"""Isolated proposal-prior diagnostic; no image, queue mutation or admission."""
from pathlib import Path
import sys,json,time,collections
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_coverage1000_queue as q
from data import action4_coverage1000_sampling as sampling
from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
from data.action4_phrase2_geometry import LocalArmVisibility
from data.action4_phrase2_collision import Phrase2CollisionChecker
from data.action4_phrase2_scene_visibility import SceneVisibility
from data.action4_coverage1000_contract import occlusion_ranges,source_eligible
cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text())
out=R/'outputs/coverage1000_v001/tail_prior_diagnostic_v001';out.mkdir(exist_ok=False)
db=q.connect();remaining=[json.loads(r[0]) for r in db.execute("SELECT spec FROM slots WHERE status IN ('pending','claimed') ORDER BY slot")];db.close()
selected=[]
for g in ['B','C','E']:selected.extend([s for s in remaining if s['group']==g][:4])
original=sampling.orientation_hint
def broad_hint(runtime,catalog,side,interval,center,rng):
    # Old-distance occlusion is a prior, not a hard physical law at a new
    # distance. The proposed widening is diagnosed, never silently adopted.
    return original(runtime,catalog,side,(0.,.75,True) if rng.random()<.7 else interval,center,rng)
rows=[];start=time.time()
for scene in ['table_full_garment','cloth_lift10_full_garment','wall']:
    slots=[s for s in selected if s['scene_id']==scene]
    if not slots:continue
    runtime=CoverageArmRuntime(scene,True,cfg['arm_asset_version']);trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']};checker=Phrase2CollisionChecker(runtime);visibility=SceneVisibility(runtime,trees)
    for slot in slots:
        for variant in ['frozen','broad_old_occlusion_prior']:
            sampling.orientation_hint=original if variant=='frozen' else broad_hint;reasons=collections.Counter();success=[]
            for n in range(32):
                attempt=1000000+n;p,error=sampling.propose(runtime,cfg,slot,attempt,catalog,trees,ray_resolution=28)
                if error:reasons[error]+=1;continue
                runtime.apply(p);collision=checker.check()
                if not collision['accepted']:reasons[collision['status']]+=1;continue
                source=visibility.measure(p);ok=True
                for side,m in source.items():
                    lo,hi,last=occlusion_ranges(slot)[side];o=m['estimated_external_occlusion'];counts=m['occluder_class_counts'];total=sum(counts.values())
                    if not max(0,lo-.04)<=o<=min(.79,hi+.04) or not source_eligible(o,m['primary_source_class']):ok=False;break
                    if o>.35 and m['primary_occluder_object']==side+'_forearm_continuation':ok=False;break
                    if total>=5 and counts.get('environment',0)+counts.get('other_actor',0)>.5*total:ok=False;break
                if not ok:reasons['actual_scene_occlusion_source']+=1;continue
                actual=runtime.export_corners();success.append({'attempt':attempt,'proposal':p,'source':source,'actual_error_px':actual['max_proposal_vs_actual_error_px']})
            rows.append({'slot_id':slot['slot_id'],'group':slot['group'],'scene':scene,'variant':variant,'attempts':32,'successes':success,'rejections':dict(reasons)})
            (out/'PROGRESS.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY_NO_RGB','rows':rows,'seconds':time.time()-start}))
sampling.orientation_hint=original
report={'status':'DIAGNOSTIC_COMPLETE_NOT_ADOPTED','rows':rows,'seconds':time.time()-start,'running_frozen_generation_unchanged':True,'queue_modified':False,'RGB_rendered':False}
(out/'COMPLETE.json').write_text(json.dumps(report));print(json.dumps({'selected_slots':len(selected),'counts':{v:sum(len(r['successes']) for r in rows if r['variant']==v) for v in ['frozen','broad_old_occlusion_prior']}}))
