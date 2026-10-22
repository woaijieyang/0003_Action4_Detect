"""Box-only development eval; score calibration never uses corner metrics."""
import json,time,os,fcntl,traceback
import torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_cached_box_quality import load_box_split
from model.action4_box_quality import BoxAwareQuality
from eval.action4_box_quality_decode import select_boxes
from eval.action4_bbox_metrics import score_boxes
from model.training.action4_runtime import atomic_json,canonical_hash

def evaluate(model,data,records):
    model.eval();predictions=[]
    with torch.inference_mode():
        for row in data:
            scores=model(row['features'].to('cuda:0'),row['geometry'].to('cuda:0')).sigmoid().cpu().numpy();predictions.append({'sample_id':row['sample_id'],'instances':select_boxes(row['candidates'],scores)})
    report=score_boxes(predictions,records,score_threshold=.3);report.update(dataset_role='development_selection_not_independent_acceptance',AP_candidate_threshold=.001,max_detections=2,candidate_pool='center_product_union',base_frozen=True)
    return report

def worker(config):
    cfg=json.loads((R/config).read_text());out=R/cfg['output'];lock=(out/'eval_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=load_box_split(R/cfg['cache_output'],'eval');records=[json.loads(l) for l in (R/'data/eval/annotations.jsonl').read_text().splitlines()];assert [x['sample_id'] for x in data]==[x['sample_id'] for x in records]
    model=BoxAwareQuality().to('cuda:0');q=out/'eval';q.mkdir(exist_ok=True);atomic_json(out/'EVAL_PROCESS.json',{'pid':os.getpid()})
    if not (out/'UNION_BASELINE.json').exists():
        predictions=[{'sample_id':x['sample_id'],'instances':select_boxes(x['candidates'],[c['old_product_score'] for c in x['candidates']])} for x in data];atomic_json(out/'UNION_BASELINE.json',score_boxes(predictions,records,score_threshold=.3))
    def rank(report):return (report['mAP50_95'],report['overall']['f1'])
    while True:
        request=None
        for p in sorted(q.glob('step_*.json')):
            if p.name.count('.')!=1 or p.with_suffix('.done.json').exists() or p.with_suffix('.failed.json').exists():continue
            try:fd=os.open(p.with_suffix('.claim'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:continue
            with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid()},f)
            request=p;break
        if request is None:
            if (out/'TRAIN_COMPLETE.json').exists() or (out/'TRAIN_FAILURE.json').exists():return
            time.sleep(3);continue
        try:
            req=json.loads(request.read_text());state=torch.load(R/req['snapshot'],map_location='cpu',weights_only=True)
            if state['schema']!='action4.box_quality_eval.v1' or state['config_id']!=canonical_hash(cfg):raise ValueError('Eval identity mismatch')
            model.load_state_dict(state['head'],strict=True);report=evaluate(model,data,records);result={'step':req['step'],'report':report};atomic_json(request.with_suffix('.done.json'),result);bp=out/'BEST.json';best=json.loads(bp.read_text()) if bp.exists() else None
            if best is None or rank(report)>rank(best['report']):
                tmp=out/'best.tmp';os.link(R/req['snapshot'],tmp);os.replace(tmp,out/'best.pt');atomic_json(bp,result)
            (R/req['snapshot']).unlink()
        except BaseException:
            atomic_json(request.with_suffix('.failed.json'),{'traceback':traceback.format_exc()});raise
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();worker(a.config)
