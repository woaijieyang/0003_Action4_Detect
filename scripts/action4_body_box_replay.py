"""Replay the published geometry layer over fixed native masks; no model calls."""
import os,sys,json,hashlib,copy,time,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
R=Path('/data/zy/vln/a-test')
def support_job(item):
    import numpy as np
    from model.action4_sam_supported_boxes import geometry_support,mask_geometry
    fid,ref=item;path=R/ref
    if path.is_symlink() or not path.resolve().is_relative_to(R):raise ValueError('Native status escaped project')
    st=json.loads(path.read_text())
    if st['global_frame_id']!=fid:raise ValueError('Frame mismatch')
    raw=R/st['raw']['arrays']
    if raw.is_symlink() or not raw.resolve().is_relative_to(R):raise ValueError('Raw source escaped project')
    with raw.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=st['raw']['sha256']:raise ValueError('Native raw changed')
    result={}
    with np.load(raw,allow_pickle=False) as z:
        ids=z['raw__outputs__out_obj_ids'].tolist();masks=z['raw__outputs__out_binary_masks']
        if len(set(ids))!=len(ids) or not set(ids)<={0,1} or masks.dtype!=np.bool_ or masks.shape!=(len(ids),2464,3248):
            raise ValueError('Native mask/ID domain differs')
        for tid,oid in [('track_00',0),('track_01',1)]:
            if oid not in ids:result[tid]={'box':None,'available':False};continue
            mask=masks[ids.index(oid)]
            clean,proof=geometry_support(mask);box,area,center=mask_geometry(clean)
            result[tid]={'box':None if box is None else box.tolist(),'available':bool(area),'cleanup':proof}
    return fid,result
def init_worker(counter,cpus):
    with counter.get_lock():index=counter.value;counter.value+=1
    os.sched_setaffinity(0,{cpus[index]})
def run():
    S=Path(__file__).resolve().parent;O=S.parent
    assert Path.cwd().resolve()==R
    sys.dont_write_bytecode=True;sys.path.insert(0,str(S/'src'))
    cfg=json.loads((S/'CONFIG.json').read_text())
    from model.action4_body_box_projection import BodyBoxProjection
    from eval.action4_sam_full_review import ReplayAudit
    parent=R/cfg['parent'];completion=json.loads((parent/'run/COMPLETE.json').read_text())
    if completion['frames']!=2850 or not completion['closed']:raise ValueError('Complete native parent required')
    src=parent/'run/FUSED_BOXES.jsonl'
    with src.open('rb') as f:source_sha=hashlib.file_digest(f,'sha256').hexdigest()
    rows=[json.loads(line) for line in src.open()]
    assert [r['frame']['global_frame_id'] for r in rows]==list(range(2850))
    if source_sha!=cfg['parent_predictions_sha256']:raise ValueError('Parent predictions changed')
    policy=json.loads((S/'POLICY.json').read_text())['policy'];projector=BodyBoxProjection(policy)
    out=O/'run';out.mkdir(exist_ok=False);audit=ReplayAudit();start=time.time()
    partial=out/'FUSED_BOXES.partial.jsonl';stats={'rejected_extent_conflict':0,'accepted_bounded_extent':0}
    ctx=multiprocessing.get_context('fork');counter=ctx.Value('i',0)
    try:
        with ProcessPoolExecutor(max_workers=len(cfg['CPU_workers']),mp_context=ctx,initializer=init_worker,initargs=(counter,cfg['CPU_workers'])) as executor:
            results=executor.map(support_job,[(i,rows[i]['native_status']) for i in range(4,2850)],chunksize=4)
            with partial.open('x') as f,(out/'SUPPORT_BOXES.jsonl').open('x') as supports:
                for i,row in enumerate(rows):
                    result=copy.deepcopy(row)
                    if i>=4:
                        fid,boxes=next(results);assert fid==i
                        supports.write(json.dumps({'frame':i,'support':boxes})+chr(10))
                        result['fused']['tracks']=[projector.update(i,row['frame']['pts']/11988,t,boxes[t['track_id']]['box']) for t in row['fused']['tracks']]
                        result['fused']['published_geometry']='bounded_body_projection_v1'
                        for t in result['fused']['tracks']:
                            assert t['candidate_id']==next(x for x in row['fused']['tracks'] if x['track_id']==t['track_id'])['candidate_id']
                            reason=t['body_box_projection'].get('enclosure_decision')
                            if reason in stats:stats[reason]+=1
                    audit.add(result);f.write(json.dumps(result,allow_nan=False)+chr(10))
                    if i%100==0:
                        (out/'PROGRESS.json').write_text(json.dumps({'frame':i,'seconds':time.time()-start,'GPU_calls':0}))
        summary=audit.finish();summary.update(native_job='outputs/action4_sam_body_box_full2850_cached_v001',parent_reanchor_run=cfg['parent'],geometry_mode='sam_primary_v1_association_plus_bounded_body_projection_v1')
        partial.replace(out/'FUSED_BOXES.jsonl')
        (out/'AUDIT.json').write_text(json.dumps(summary,indent=2))
        (out/'COMPLETE.json').write_text(json.dumps({'status':'FULL2850_BODY_BOX_PROJECTION_CANDIDATE_NOT_ACCURACY','frames':2850,'source_predictions':{'path':str(src.relative_to(R)),'sha256':source_sha},'events':completion['events'],'events_origin':'unchanged parent native reanchor run','native_calls':0,'GPU_calls':0,'association_and_native_generation_unchanged':True,'counts':stats,'seconds':time.time()-start,'accuracy_accepted':False},indent=2))
    except BaseException:
        import traceback
        (out/'FAILED.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
if __name__=='__main__':run()
