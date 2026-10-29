"""Actual PNG/NPZ acceptance, distinct from the later human visual gate."""
from pathlib import Path
import sys,os,json,time,traceback,shutil
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data import action4_coverage1000_queue as queue
from data.action4_coverage1000_quality import validate,publish
from data.action4_phrase2_host_worker import surface_check


def remove_owned(payload,cfg):
    p=contained_path(R,payload['staging_dir'],exists=True)
    parent=R/cfg['candidate_root']/payload['slot']['split']/'.staging'/payload['slot']['sample_id']
    if p.parent!=parent or p.is_symlink():raise ValueError('Not an owned attempt directory')
    for root,dirs,files in os.walk(p,followlinks=False):
        if any((Path(root)/n).is_symlink() for n in dirs+files):raise ValueError('Unexpected candidate link')
    shutil.rmtree(p)


def run(name):
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());out=R/cfg['output_root'];directory=out/'quality_workers'/name
    directory.mkdir(parents=True,exist_ok=False);processed=0
    while not(out/'STOP.json').exists():
        claim=queue.claim(quality=True)
        if claim is None:
            if not queue.pending_enabled():break
            time.sleep(1);continue
        slot=json.loads(claim['spec'])
        try:
            pointer=json.loads(claim['record'])['candidate'];payload=json.loads(contained_path(R,pointer,exists=True).read_text())
            if payload['slot']!=slot or not payload['collision']['accepted'] or payload['actual_projection_error_px']>.005:raise ValueError('Immutable frame/slot binding failed')
            if payload['record'].get('output_role')!='dataset_candidate':raise ValueError('A diagnostic RGB may never enter the dataset queue')
            ok,reason,stats=validate(payload['record'],payload['proposal'],slot,cfg)
            if ok:
                geometry=surface_check(payload['record'])
                if not(out/'ROOT_PREVIEW_ACCEPTED.json').exists():
                    queue.transition(slot['slot_id'],'preview_checked',record={'candidate':pointer,'stats':stats,'surface_validation':geometry})
                    (R/payload['runtime_dir']/'PREVIEW_CHECKED.json').write_text(json.dumps({'actual':stats,'geometry':geometry,'formal_published':False},indent=2))
                else:
                    gate=json.loads((out/'ROOT_PREVIEW_ACCEPTED.json').read_text())
                    if gate.get('status')!='PASS_FORMAL128_VISUAL_REVIEW' or gate.get('protocol_id')!=cfg['protocol_id']:raise ValueError('Missing or stale formal-preview visual admission')
                    accepted=publish(payload['record'],payload['proposal'],slot,stats,cfg);accepted.update(surface_validation=geometry,source_candidate_record=pointer)
                    queue.transition(slot['slot_id'],'accepted',record=accepted)
                    (R/payload['runtime_dir']/'ACCEPTED.json').write_text(json.dumps({'actual':stats,'geometry':geometry}))
                    remove_owned(payload,cfg)
            else:
                db=queue.connect();db.execute('CREATE TABLE IF NOT EXISTS failures(reason TEXT PRIMARY KEY,count INTEGER)')
                db.execute('INSERT INTO failures VALUES(?,1) ON CONFLICT(reason) DO UPDATE SET count=count+1',(reason,));n=db.execute('SELECT count FROM failures WHERE reason=?',(reason,)).fetchone()[0];db.close()
                (R/payload['runtime_dir']/'REJECTED.json').write_text(json.dumps({'reason':reason,'stats':stats,'slot':slot,'attempt':payload['proposal']['phrase2']['attempt']}))
                if n>3:remove_owned(payload,cfg)
                queue.transition(slot['slot_id'],'pending',error=reason)
            processed+=1;(directory/'STATUS.json').write_text(json.dumps({'pid':os.getpid(),'processed':processed}))
        except BaseException:
            failure={'pid':os.getpid(),'slot_id':slot['slot_id'],'traceback':traceback.format_exc()};(directory/'FAILURE.json').write_text(json.dumps(failure,indent=2));(out/'STOP.json').write_text(json.dumps(failure,indent=2));queue.transition(slot['slot_id'],'failed',error=failure['traceback']);raise
    (directory/'COMPLETE.json').write_text(json.dumps({'processed':processed}))

if __name__=='__main__':run(sys.argv[1])
