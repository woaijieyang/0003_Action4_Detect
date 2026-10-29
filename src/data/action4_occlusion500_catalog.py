"""Add split-local, already geometry-admitted orientation calibration examples.

Only roll/direction and sightline calibrations are retained. Size, full body
translation, arm-pair transforms, RGB and source random seeds are not reused.
Calibration poses never cross train/eval/test. Final rays/masks remain required.
"""
from pathlib import Path
import json,copy,time,sys,collections,hashlib
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def build(output,require_complete=True):
    from data import action4_phrase2_queue as q
    from data.action4_occlusion500_contract import occlusion_bin,source_eligible
    db=q.connect();rows=db.execute("SELECT record FROM slots WHERE status='accepted' ORDER BY slot").fetchall();db.close()
    records=[json.loads(r[0]) for r in rows]
    if require_complete and len(records)!=5000:raise ValueError('Wait for all5000 before freezing the additional production catalog')
    raw=json.loads((R/'outputs/phrase2_generation_v001/OCCLUSION_CATALOG_v006.json').read_text());result={'schema':'action4.occlusion500_split_local_calibration.v1','by_split':{},'base_accepted_snapshot':len(records),'source_policy':'Same split only. Original coarse calibration plus admitted roll/direction/sightline; all continuous parameters resampled.','time_unix':time.time()}
    counts={}
    for split in ['train','eval','test']:
        catalog=copy.deepcopy(raw);counts[split]={s:[0]*5 for s in ['left','right']}
        for r in records:
            if r['split']!=split:continue
            for obj,side in zip(r['instances'],['left','right']):
                kind=occlusion_bin(obj['occlusion_inframe']);source=r['occlusion_sources'][side]
                if kind is None or not source_eligible(obj['occlusion_inframe'],source['primary_source_class']):continue
                note=r['phrase2']['by_side'][side];direction=note['forearm_direction_capture'];center=note['bbox_center_uv']
                anchors=catalog['rows'][side]['anchors'];a=min(anchors,key=lambda a:sum((x-y)**2 for x,y in zip(a['center_uv'],center)))
                a['samples'].append({'roll_deg':note['forearm_roll_deg'],'direction_x':direction[0],'direction_z':direction[2],
                    'occlusion_estimate':obj['occlusion_inframe'],'primary_class':'own_wrist_forearm','calibration_from':'admitted_mask_and_pose',
                    'calibration_split':split,'calibration_scene':r['scene_id'],'source_slot_id':r['slot_id']})
                counts[split][side][kind]+=1
        result['by_split'][split]=catalog
    result['added_orientation_counts']=counts
    Path(output).open('x').write(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--partial-diagnostic',action='store_true');a=p.parse_args()
    target=(R/a.output).resolve();assert target.is_relative_to(R/'outputs/phrase2_occlusion500_v001')
    result=build(target,not a.partial_diagnostic);print(json.dumps({'records':result['base_accepted_snapshot'],'added_orientation_counts':result['added_orientation_counts']}))
