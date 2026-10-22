"""Finish a known paired solver run and retain every output discrepancy."""
from pathlib import Path
import json,time,subprocess
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json


def compare(reference,candidate):
    reference=contained_path(R,reference,exists=True);candidate=contained_path(R,candidate,exists=True)
    results={};passed=True
    for variant in ['raw','rigid','visibility_residual']:
        a=[json.loads(l) for l in (reference/(variant+'_predictions.jsonl')).read_text().splitlines()]
        b=[json.loads(l) for l in (candidate/(variant+'_predictions.jsonl')).read_text().splitlines()]
        ids_a=[x['sample_id'] for x in a];ids_b=[x['sample_id'] for x in b];same_ids=ids_a==ids_b and len(set(ids_a))==len(a)==200
        differences=[];structure=[]
        if same_ids:
            for x,y in zip(a,b):
                if len(x['instances'])!=len(y['instances']):structure.append(x['sample_id']);continue
                for u,v in zip(x['instances'],y['instances']):
                    for k in ['instance_id','object_id','corner_ids','bbox_full_xyxy','score']:
                        if u.get(k)!=v.get(k):structure.append(x['sample_id']+'/'+k)
                    delta=np.abs(np.array(u['corners_px'])-np.array(v['corners_px']));differences.extend(delta.flatten().tolist())
        maximum=float(np.max(differences)) if differences else None
        equal=bool(same_ids and not structure and maximum is not None and np.isfinite(differences).all() and maximum<.001)
        results[variant]={'images':len(a),'same_ordered_unique_ids':same_ids,'structural_differences':structure,'max_native_coordinate_delta_px':maximum,'p95_delta_px':float(np.percentile(differences,95)) if differences else None,'within_0_001px':equal}
        passed=passed and equal
    ar=json.loads((reference/'REPORT.json').read_text());br=json.loads((candidate/'REPORT.json').read_text());metrics={}
    for variant in results:
        metrics[variant]={}
        for threshold in ar['scores'][variant]['thresholds']:
            a=ar['scores'][variant]['thresholds'][threshold];b=br['scores'][variant]['thresholds'][threshold]
            keys=['tp','fp','fn','valid_gt','predictions','f1','precision','recall']
            matched=all(a[k]==b[k] for k in keys);passed=passed and matched
            metrics[variant][threshold]={'identical_counts_and_rates':matched,'reference':{k:a[k] for k in keys},'candidate':{k:b[k] for k in keys}}
    report={'status':'PAIRED_EQUIVALENCE_PASS' if passed else 'PAIRED_EQUIVALENCE_DIFFERENCES','variants':results,'metrics':metrics,'scope':'full200 development solver equivalence, not real accuracy acceptance','current_video_changed':False,'production_accepted':False}
    atomic_json(candidate/'EQUIVALENCE.json',report)


def wait_then_compare(reference,candidate,pid):
    candidate=contained_path(R,candidate,exists=True)
    if (candidate/'EQUIVALENCE.json').exists():raise FileExistsError('Already compared')
    while not (candidate/'REPORT.json').exists():
        state=subprocess.run(['ps','-p',str(pid),'-o','stat=,args='],capture_output=True,text=True)
        if state.returncode!=0 or state.stdout.strip().startswith('Z'):
            atomic_json(candidate/'POSTPROCESS_FAILURE.json',{'reason':'source process exited before REPORT; no restart','pid':pid});return
        if 'eval.action4_geometry_comparison' not in state.stdout or str(candidate.relative_to(R)) not in state.stdout:raise RuntimeError('PID command identity changed')
        time.sleep(30)
    compare(reference,candidate)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args();wait_then_compare(a.reference,a.candidate,a.pid)
