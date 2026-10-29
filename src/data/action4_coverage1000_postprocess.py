"""Wait without restarts, then run bounded data-only audits once."""
from pathlib import Path
import json,os,subprocess,sys,time,fcntl
from data.action4_paths import PROJECT_ROOT as R

def run():
    out=R/'outputs/coverage1000_v001';state=out/'POSTPROCESS_PID.json'
    if state.exists():raise FileExistsError('Do not launch duplicate postprocessing')
    state.write_text(json.dumps({'pid':os.getpid(),'started_unix':time.time()},indent=2))
    while True:
        p=out/'PIPELINE_STATUS.json'
        if p.exists():
            status=json.loads(p.read_text())
            if status['status']=='ALL1000_ENGINEERING_ACCEPTED_GLOBAL_AUDITS_REQUIRED':break
            if status['status'] in ['FAILED_REQUIRES_DIAGNOSIS','INCOMPLETE_NO_AUTOMATIC_RESTART']:raise RuntimeError('Generation did not finish; preserve failure, do not restart')
        time.sleep(60)
    guard=(out/'controller.lock').open('a')
    for retry in range(10):
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if retry==9:raise
            time.sleep(1)
    env=os.environ.copy();env.update(PYTHONPATH=str(R/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'runtime'))
    steps=[('audit','data.action4_coverage1000_audit',[],[22,23]),
        ('similarity','data.action4_coverage1000_similarity',['--workers','64'],list(range(48))+list(range(72,96))),
        ('candidate_interface','data.action4_coverage1000_interface',[],[22,23])]
    for name,module,args,cpus in steps:
        receipt=out/f'POSTPROCESS_{name.upper()}_COMPLETE.json'
        if receipt.exists():continue
        cmd=['taskset','-c',','.join(map(str,cpus)),str(R/'.venv/bin/python'),'-B','-m',module]+args
        with(out/f'POSTPROCESS_{name}.log').open('xb') as f:
            p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
            (out/'POSTPROCESS_STATUS.json').write_text(json.dumps({'status':'RUNNING','step':name,'pid':p.pid,'command':cmd}))
            result=p.wait()
        if result:raise RuntimeError('Postprocess step failed: '+name+'; read its log without weakening checks')
        if name=='audit':
            report=json.loads((out/'FINAL_AUDIT.json').read_text())
            if report['status']!='PASS_ACTUAL1000_QUOTA_FILE_AND_HOST_GEOMETRY_CLOSURE' or report['images']!=1000:raise ValueError('Audit did not establish actual1000 closure')
        if name=='candidate_interface':
            report=json.loads((out/'DATA_INTERFACE_VALIDATION.json').read_text())
            if report['status']!='PASS_ACTUAL_DATA_BATCHES_NO_TRAINING' or report['all_detector_targets_checked']!=2000:raise ValueError('Actual candidate item/target chain not validated')
        receipt.write_text(json.dumps({'status':'COMMAND_COMPLETED_REVIEW_RESULT','command':cmd,'time_unix':time.time()}))
        if name=='similarity':
            proof=json.loads((out/'SIMILARITY_CURRENT.json').read_text())
            if proof['status']!='PASS_GLOBAL_SIMILARITY_AUDIT':
                (out/'POSTPROCESS_STATUS.json').write_text(json.dumps({'status':'RESAMPLE_REQUIRED','rejected':len(proof['rejected']),'report':'outputs/coverage1000_v001/SIMILARITY_CURRENT.json'}));return
    cmd=['taskset','-c','22,23',str(R/'.venv/bin/python'),'-B','-m','data.action4_coverage1000_final_review']
    with(out/'POSTPROCESS_final_review.log').open('xb') as f:p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);result=p.wait()
    if result:raise RuntimeError('Final representative input export failed; inspect log')
    (out/'POSTPROCESS_STATUS.json').write_text(json.dumps({'status':'AWAIT_ROOT_FINAL_VISUAL_REVIEW','bundle':'outputs/coverage1000_v001/final_review_v001/FINAL_REVIEW_BUNDLE.json'}))

if __name__=='__main__':run()
