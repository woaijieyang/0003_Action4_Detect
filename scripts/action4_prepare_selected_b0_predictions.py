"""Frozen completed-B0 train predictions; reuse existing complete eval predictions.

No GT bbox/mask/pose is an inference argument. No SAM calls or model promotion.
"""
import os,sys,json,hashlib,time,subprocess,argparse,fcntl
from pathlib import Path
R=Path('/data/zy/vln/a-test')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if Path.cwd().resolve()!=R or a.gpu not in (4,5,6,7):raise ValueError('Authorized project/GPU required')
 out=(R/a.output).resolve();assert out.is_relative_to(R/'outputs')
 if out.exists() and not a.resume:raise FileExistsError('Existing producer output; explicit verified resume required')
 out.mkdir(parents=True,exist_ok=True);guard=(out/'producer.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 seal_path=R/'models/candidates/action4_improvement/stage1_B0_completed_20261024/SEALED_ROLE.json';seal=json.loads(seal_path.read_text())
 if seal['role']!='B0' or not seal['strict_load'] or seal['epochs']!=80 or seal['completed_optimizer_steps']!=13040 or seal['completion']['evaluations']!=653:raise ValueError('Incomplete selectedB0 release')
 spec_path=R/seal['run_spec'];spec=json.loads(spec_path.read_text());code=R/spec['code_root'];cfg=json.loads((R/spec['config']).read_text())
 if sha(spec_path)!=seal['run_spec_sha256'] or sha(R/seal['bundle'])!=seal['sha256']:raise ValueError('Selected model/source changed')
 active=json.loads((R/'configs/execution/action4_active.json').read_text())['active_ref'];authority_path=(R/active['path']).resolve()
 if not authority_path.is_relative_to(R) or sha(authority_path)!=active['sha256']:raise ValueError('Resource authority mismatch')
 authority=json.loads(authority_path.read_text())
 if a.gpu not in authority['authorized_evaluation_gpu'] or a.gpu in authority['forbidden_gpu_ids']:raise ValueError('GPU permission revoked')
 if not set(os.sched_getaffinity(0))<=set(authority['cpu_policy']['allowed_logical_cpu_ids']) or len(os.sched_getaffinity(0))>6:raise ValueError('Bounded tail CPU allocation required')
 gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu[0].strip() or int(gpu[1])<12288:raise ValueError('ExactGPU mapping and12GiB headroom required')
 os.environ['ACTION4_PROJECT_ROOT']=str(R);sys.path[:0]=[str(code/'runtime/surface_deps'),str(code/'src')]
 from model.training.action4_run_snapshot import verify_running_snapshot
 verify_running_snapshot(cfg)
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
 from model.action4_release_bundle import load_bundle
 from eval.action4_b0_scoring import predict
 from eval.action4_prefetch import ordered_items
 from data.action4_component_data import ComponentData
 from data.action4_dataset import Action4Dataset
 model,_,state=load_bundle(seal['bundle'],'B0','cuda:0');assert state['selected_step']==seal['selected_step'] and state['config_id']==seal['config_id']
 data=ComponentData(seal['validation_receipt'],'train','B0');assert len(data)==5200
 manifest=json.loads((R/seal['validation_receipt']).read_text())['splits'];ids=[row['sample_id'] for row in data.records];assert len(set(ids))==5200
 identity={'schema':'action4.selected_B0_prediction_producer.v1','B0_seal':{'path':str(seal_path.relative_to(R)),'sha256':sha(seal_path)},'run_spec':{'path':seal['run_spec'],'sha256':sha(spec_path)},'B0_bundle':{'path':seal['bundle'],'sha256':seal['sha256']},'selected_step':seal['selected_step'],'config_id':seal['config_id'],'source_code_sha256':spec['project_code_sha256'],'producer_sha256':sha(Path(__file__)),
   'data_manifest_sha256':{k:manifest[k]['sha256'] for k in ('train','eval')},'RGB_variant':'native_original','GT_used_for_inference':False,'scoring_policy':cfg['evaluation_scoring_policy'],'publication_threshold':.3}
 ip=out/'IDENTITY.json'
 if ip.exists():
  if json.loads(ip.read_text())!=identity:raise ValueError('Resume producer identity mismatch')
 else:atomic(ip,identity)
 rows=[];journal=out/'train_predictions.jsonl'
 if journal.exists():
  for i,line in enumerate(journal.read_text().splitlines()):
   row=json.loads(line)
   if row['sample_id']!=ids[i]:raise ValueError('Resume row order/identity mismatch')
   record=data.records[i]
   if tuple(data.stats[record['image']])!=Action4Dataset._file_identity(R/record['image']):raise ValueError('Resume input file changed')
   rows.append(row)
 start=len(rows)
 class Remaining:
  def __len__(self):return len(data)-start
  def __getitem__(self,index):return data[index+start]
 started=time.time()
 with journal.open('a') as stream:
  for _,item in ordered_items(Remaining(),workers=4,ahead=8):
   record=item['record']
   if sha(R/record['image'])!=record['rgb_sha256']:raise ValueError('Actual RGB source changed')
   with torch.inference_mode():instances=predict(model,item['image'],state['config'],'cuda:0',cfg['evaluation_scoring_policy'])
   row={'sample_id':record['sample_id'],'instances':instances};stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
   if len(rows)%25==0 or len(rows)==5200:atomic(out/'STATUS.json',{'status':'PREDICTING_TRAIN','completed':len(rows),'expected':5200,'elapsed_seconds':time.time()-started,'GPU':a.gpu})
 if len(rows)!=5200 or [row['sample_id'] for row in rows]!=ids:raise ValueError('Incomplete train prediction domain')
 train=out/'TRAIN_PREDICTIONS.json';atomic(train,rows);train.chmod(0o444)
 best=json.loads((R/cfg['output']/'BEST.json').read_text());report=R/best['raw_report'];evpred=report.parent/'PREDICTIONS.json';ev=json.loads(evpred.read_text());eval_data=ComponentData(seal['validation_receipt'],'eval','B0')
 if len(ev)!=650 or [x['sample_id'] for x in ev]!=[x['sample_id'] for x in eval_data.records]:raise ValueError('Existing selectedB0 eval prediction coverage mismatch')
 receipt={'status':'COMPLETE_SELECTED_B0_TRAIN_AND_REUSED_EVAL','identity':identity,'train_images':5200,'train_predictions':{'path':str(train.relative_to(R)),'sha256':sha(train)},'eval_images':650,'eval_predictions':{'path':str(evpred.relative_to(R)),'sha256':sha(evpred)},'eval_reinferred':False,'new_native_B0_images':5200-start,'train_saved_candidates':sum(len(r['instances']) for r in rows),'eval_saved_candidates':sum(len(r['instances']) for r in ev),'SAM_calls':0,'default_weights_modified':False}
 atomic(out/'COMPLETE.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='identity'}),flush=True)
if __name__=='__main__':main()
