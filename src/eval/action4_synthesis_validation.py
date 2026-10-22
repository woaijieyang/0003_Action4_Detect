"""Parallel actual synthetic-data validation using existing geometry/mask and training consumers.
No scene construction, rendering, training, source publication or acceptance mutation.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
ROOT=Path('/data/zy/vln/a-test')
AID_REF={'path':'src/eval/action4_corner_checks.py','sha256':'c847e052669fc90ac42cc37ba0391d55201b34e8fa246341fce3c2cdf82d5427'}
CONSUMER_SOURCES=('src/data/action4_dataset.py','src/data/action4_adapter.py','src/data/action4_paths.py','src/model/training/action4_trainer.py','src/model/training/action4_runtime.py','src/eval/action4_runner.py','src/eval/action4_metrics.py','src/model/action4_decode.py','src/model/action4_losses.py','src/eval/action4_corner_checks.py')
def bounded(v):
 p=Path(v)
 if '..' in p.parts:raise ValueError('Path traversal')
 p=(p if p.is_absolute() else ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
 return p
def ref(v):
 p=bounded(v);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def readref(r):
 p=bounded(r['path']);raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Input identity changed: '+r['path'])
 return json.loads(raw)
def write_new(p,v):
 with bounded(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load_aid():
 p=bounded(AID_REF['path'])
 if ref(p)!=AID_REF:raise ValueError('Original reviewed geometry/mask aid changed')
 from eval import action4_corner_checks as m
 return m
def initialize_worker(threads):
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]=str(threads)
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HIP_VISIBLE_DEVICES']='';os.environ['NVIDIA_VISIBLE_DEVICES']='void'
 import torch
 torch.set_num_threads(threads)

def input_snapshot(plan_ref,data_root,pattern,count):
 from data.action4_synthesis import select_shard_entries,_run_output_dir
 aid=load_aid();reader=aid.ReadSet();plan=reader.json(plan_ref['path'],plan_ref['sha256'])
 if type(count) is not int or not 1<=count<=plan['count']:raise ValueError('Exact positive expected prefix count')
 expected=plan['entries'][:count];expected_ids=[e['sample_id'] for e in expected];seen=set();jobs=[]
 root=bounded(data_root)
 if not root.is_relative_to(ROOT/'data'):raise ValueError('Read data only from project data directory')
 if '/' in pattern or '..' in pattern:raise ValueError('Direct child shard pattern only')
 for data in sorted(root.glob(pattern)):
  data=bounded(data)
  if not data.is_dir():continue
  run=bounded(_run_output_dir(data));launch=reader.json(run/'launch.json');completion=reader.json(run/'render_completion.json');host=reader.json(run/'geometry_validation.json');raw=reader.data(data/'manifest.jsonl');rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
  mref={'path':str((data/'manifest.jsonl').relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()};command=launch['command']
  if launch.get('plan_sha256')!=plan_ref['sha256'] or aid.rel(aid.command_value(command,'--plan',''))!=plan_ref['path'] or aid.rel(aid.command_value(command,'--output',''))!=str(data.relative_to(ROOT)):raise ValueError('Launch plan/output mismatch')
  si=int(aid.command_value(command,'--shard-index','0'));ns=int(aid.command_value(command,'--num-shards','1'));start=int(aid.command_value(command,'--entry-start','0'));n=aid.command_value(command,'--entry-count');limit=aid.command_value(command,'--limit')
  if ns<1 or not 0<=si<ns or start<0 or n is not None and limit is not None:raise ValueError('Invalid launch partition')
  n=int(n if n is not None else limit) if n is not None or limit is not None else None
  chosen=select_shard_entries(plan,si,ns)[start:None if n is None else start+n];ids=[e['sample_id'] for e in chosen]
  if not ids or [r['sample_id'] for r in rows]!=ids or launch['expected_sample_ids']!=ids or completion['sample_ids']!=ids or host['sample_ids']!=ids:raise ValueError('Complete ordered manifest/launch/Host binding required')
  closing=completion['worker_batch_closing']
  if completion.get('status')!='rendered_pending_independent_validation' or completion.get('manifest_ref')!=mref or completion.get('training_accepted') is not False or closing.get('status')!='PASS_WORKER_METADATA_CLOSED_MANIFEST_NOT_YET_PUBLISHED' or closing.get('plan_ref')!=plan_ref or closing.get('dependency_map_unchanged') is not True or closing.get('completed_sample_ids')!=sorted(ids):raise ValueError('Original worker closing required')
  if host.get('status')!='geometry_verified_visual_acceptance_pending' or host.get('training_accepted') is not False or host.get('manifest_sha256')!=mref['sha256'] or host.get('images')!=len(rows) or host.get('instances')!=sum(len(r['instances']) for r in rows) or host.get('corners')!=8*host['instances']:raise ValueError('Bound complete Host geometry receipt required')
  for row,e in zip(rows,chosen):
   aid.check_presence(row,e)
   if row['sample_id'] in seen or row.get('status')!='render_candidate_not_training_accepted' or row.get('training_accepted') is not False or row.get('dataset_generation_allowed') is not False or row.get('asset_backend')!='modular_composed' or row['pose_controls_proof']['plan_ref']!=plan_ref or row['modular_backend']['plan_sha256']!=plan_ref['sha256']:raise ValueError('Candidate identity/status/plan mismatch')
   seen.add(row['sample_id'])
  jobs.append({'data_dir':str(data.relative_to(ROOT)),'manifest_ref':mref,'entries':chosen,'host_ref':{'path':str((run/'geometry_validation.json').relative_to(ROOT)),'sha256':reader.refs[str((run/'geometry_validation.json').relative_to(ROOT))]}})
 if seen!=set(expected_ids):raise ValueError('Missing/extra sample IDs; exact requested prefix required')
 return plan,reader,jobs

def consumer_check(item,config):
 import numpy as np
 import torch
 from data.action4_dataset import prepare_full_images,prepare_rois,build_detection_targets,build_corner_targets,is_ignored
 from data.action4_adapter import DataAdapter
 from model.training.action4_trainer import _roi_builder
 from eval.action4_runner import heatmap_to_original
 from model.action4_decode import decode_corners
 row=item['record'];image=item['image'];instances=[x for x in row['instances'] if not is_ignored(x)];N=len(instances)
 _,full_a=prepare_full_images([image],size=tuple(config['preprocess']['full_size']))
 detections=build_detection_targets([row],full_a,full_size=tuple(config['preprocess']['full_size']));assert int(detections['instance_mask'].sum())==N
 rois,t=_roi_builder([image],[row],config,'cpu',np.random.default_rng(0),jitter=False)
 size=config['preprocess']['roi_size'];hs=size//4;assert rois.shape==(N,3,size,size) and t['targets'].shape==(N,8,hs,hs) and t['valid'].shape==(N,8) and torch.isfinite(t['targets']).all()
 indices=[j for j,x in enumerate(row['instances']) if not is_ignored(x)]
 _,aff=prepare_rois([image],[x['bbox_full_xyxy'] for x in instances],[0]*N,roi_size=size,expansion=config['preprocess']['roi_expansion'])
 maps,valid=build_corner_targets([row],[0]*N,indices,aff,roi_size=size,heatmap_size=hs);assert torch.equal(maps,t['targets']) and torch.equal(valid,t['valid'])
 inverses=heatmap_to_original(aff,size,(hs,hs));down,_=DataAdapter.letterbox_affine((size,size),(hs,hs));max_error=0.;quant=0.;hidden=0;outside=0
 for j,instance in enumerate(instances):
  points=np.asarray(instance['corners_px'],float);assert points.shape==(8,2);assert np.allclose(np.r_[points.min(0),points.max(0)],instance['bbox_full_xyxy'],atol=.003,rtol=0)
  hm=DataAdapter.transform_points(points,down@aff[j]);back=DataAdapter.transform_points(hm,inverses[j]);error=float(np.max(np.linalg.norm(back-points,axis=1)));assert error<1e-6;max_error=max(max_error,error)
  fullpoints=DataAdapter.transform_points(points,full_a[0]);assert np.max(np.abs(DataAdapter.transform_points(fullpoints,full_a[0],inverse=True)-points))<1e-6
  support=np.all((hm>=0)&(hm<=hs-1),axis=1);assert np.array_equal(valid[j].numpy(),support)
  decoded=decode_corners(maps[j:j+1],topk=1,from_logits=False)[0].numpy();errors=np.linalg.norm(decoded[support]-hm[support],axis=1)
  if len(errors):quant=max(quant,float(errors.max()));assert errors.max()<=np.sqrt(.5)+1e-5
  assert len(instance['corner_visible'])==8;hidden+=sum(not bool(x) for x in instance['corner_visible']);outside+=int(((points[:,0]<0)|(points[:,0]>=image.shape[2])|(points[:,1]<0)|(points[:,1]>=image.shape[1])).sum())
 return {'physical_ids':[x['instance_id'] for x in instances],'roi_count':N,'supported_corner_count':int(valid.sum()),'max_native_round_trip_error_px':max_error,'target_top1_quantization_error_hm_px':quant,'corner_visible_false_count':hidden,'native_outside_corner_count':outside,'full_bbox_from_all8':True,'model_predictions_tested':False}

def check_shard(job,out,config_ref,module_ref):
 from PIL import Image,ImageDraw
 import numpy as np
 from data.action4_dataset import Action4Dataset
 start=time.perf_counter();worker_out=bounded(out)/('shard_'+str(job['ordinal']).zfill(2));worker_out.mkdir()
 try:
  if ref(__file__)!=module_ref or any(ref(r['path'])!=r for r in job['consumer_source_refs']):raise ValueError('Validator/consumer source drift')
  aid=load_aid();reader=aid.ReadSet()
  for path,h in aid.DEPENDENCIES.items():reader.data(path,h)
  raw=reader.data(job['manifest_ref']['path'],job['manifest_ref']['sha256']);rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()];config=readref(config_ref)
  ds=Action4Dataset(ROOT,job['manifest_ref']['path'],allow_diagnostic=True)
  if ds.records!=rows:raise ValueError('Dataset loaded different records')
  try:Action4Dataset(ROOT,job['manifest_ref']['path'])
  except ValueError as e:
   if 'unaccepted' not in str(e):raise
  else:raise AssertionError('Formal training admitted unaccepted candidates')
  results=[]
  for n,(row,entry) in enumerate(zip(rows,job['entries'])):
   aid.check_presence(row,entry);overlay,instances,semantics=aid.validate_record(reader,row,bounded(job['data_dir']));item=ds.collate_fn([ds[n]])[0];metrics=consumer_check(item,config)
   sid=row['sample_id']
   if not re.fullmatch(r'[A-Za-z0-9_-]+',sid):raise ValueError('Unsafe sample filename')
   overlay_path=worker_out/(sid+'_overlay.png');overlay.save(overlay_path)
   tile=Image.new('RGB',(480,600),(25,25,25));thumb=overlay.copy();thumb.thumbnail((480,365));tile.paste(thumb,((480-thumb.width)//2,0));draw=ImageDraw.Draw(tile);draw.text((4,367),sid+' '+entry['scene_family'],fill='white')
   for j,instance in enumerate(instances):
    uv=np.asarray(instance['independent_points_native_px']);low=uv.min(0);high=uv.max(0);box=(max(0,int(np.floor(low[0]))-32),max(0,int(np.floor(low[1]))-32),min(overlay.width,int(np.ceil(high[0]))+33),min(overlay.height,int(np.ceil(high[1]))+33));assert box[2]>box[0] and box[3]>box[1]
    crop=overlay.crop(box);crop.thumbnail((238,195));tile.paste(crop,(j*240+(238-crop.width)//2,390));draw.text((j*240+4,584),'physical ID '+str(instance['instance_id']),fill='white')
   tile_path=worker_out/(sid+'_tile.png');tile.save(tile_path)
   results.append({'sample_id':sid,'entry':entry,'instances':instances,'mask_semantics':semantics,'consumer':metrics,'overlay':str(overlay_path.relative_to(ROOT)),'tile':str(tile_path.relative_to(ROOT))})
  identities=reader.close()
  if ref(__file__)!=module_ref or ref(config_ref['path'])!=config_ref or any(ref(r['path'])!=r for r in job['consumer_source_refs']):raise ValueError('Consumer/config source changed')
  report={'status':'PASS_SHARD_POINTS_MASKS_CONSUMER_VISUAL_PENDING','manifest_ref':job['manifest_ref'],'samples':results,'input_refs':identities,'consumer_source_refs':job['consumer_source_refs'],'elapsed_seconds':time.perf_counter()-start,'training_accepted':False};write_new(worker_out/'REPORT.json',report);return report
 except BaseException as e:
  write_new(worker_out/'FAILED.json',{'error':repr(e),'traceback':traceback.format_exc(),'training_accepted':False});raise

def run(plan_ref,data_root,pattern,out,count,workers,threads,cpu_budget,config_path):
 if not __debug__:raise RuntimeError('Validation cannot run with Python assertion optimization')
 if Path.cwd().resolve()!=ROOT:raise ValueError('Exact project root required')
 if not 1<=cpu_budget<=72 or not 1<=threads<=8 or workers<1:raise ValueError('Bounded explicit CPU budget')
 start=time.perf_counter();plan,reader,jobs=input_snapshot(plan_ref,data_root,pattern,count);config_ref=ref(config_path);module_ref=ref(__file__);consumer_source_refs=[ref(p) for p in CONSUMER_SOURCES]
 output=bounded(out)
 if output==ROOT/'outputs' or not output.is_relative_to(ROOT/'outputs') or output.exists():raise ValueError('Fresh project outputs directory required')
 output.mkdir(parents=True);workers=min(workers,len(jobs),cpu_budget//threads)
 if workers<1:raise ValueError('Insufficient declared CPU budget')
 write_new(output/'STARTED.json',{'plan_ref':plan_ref,'count':count,'shards':len(jobs),'workers':workers,'threads_each':threads,'CPU_budget':cpu_budget,'module_ref':module_ref,'original_aid_ref':AID_REF,'config_ref':config_ref,'consumer_source_refs':consumer_source_refs,'parent_pid':os.getpid(),'scope':'Reuse completed Host; actual image/point/mask and real Dataset/train/eval transforms; no model or training acceptance'})
 results=[]
 try:
  with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(threads,)) as pool:
   futures=[pool.submit(check_shard,{**j,'ordinal':i,'consumer_source_refs':consumer_source_refs},str(output),config_ref,module_ref) for i,j in enumerate(jobs)]
   for future in as_completed(futures):
    result=future.result();results.extend(result['samples']);print('VALIDATED',len(results),'/',count,flush=True)
  by_id={r['sample_id']:r for r in results};expected=plan['entries'][:count]
  if len(results)!=count or len(by_id)!=count or set(by_id)!={e['sample_id'] for e in expected}:raise ValueError('Incomplete/duplicate parallel results')
  results=[by_id[e['sample_id']] for e in expected];parent_refs=reader.close()
  if ref(__file__)!=module_ref or ref(config_ref['path'])!=config_ref or any(ref(r['path'])!=r for r in consumer_source_refs):raise ValueError('Pipeline source/config drift')
  from PIL import Image
  sheets=[]
  for begin in range(0,count,16):
   batch=results[begin:begin+16];sheet=Image.new('RGB',(1920,600*((len(batch)+3)//4)),(25,25,25))
   for i,row in enumerate(batch):
    with Image.open(bounded(row['tile'])) as tile:sheet.paste(tile,((i%4)*480,(i//4)*600))
   name='sheet_'+str(len(sheets)+1).zfill(2)+'.jpg';sheet.save(output/name,quality=94);sheets.append(name)
  index={'plan_ref':plan_ref,'sample_ids':[e['sample_id'] for e in expected],'manifest_refs':[j['manifest_ref'] for j in jobs]};write_new(output/'MANIFEST_INDEX.json',index)
  report={'status':'PASS_POINTS_MASKS_DATAFLOW_ROOT_VISUAL_PENDING','plan_ref':plan_ref,'count':count,'instance_count':sum(len(r['instances']) for r in results),'samples':results,'split_counts':{s:sum(r['entry']['split']==s for r in results) for s in ('train','eval','test')},'sheets':sheets,'manifest_refs':index['manifest_refs'],'host_refs':[j['host_ref'] for j in jobs],'parent_input_refs':parent_refs,'module_ref':module_ref,'config_ref':config_ref,'consumer_source_refs':consumer_source_refs,'workers':workers,'threads_each':threads,'elapsed_seconds':time.perf_counter()-start,'max_native_round_trip_error_px':max(r['consumer']['max_native_round_trip_error_px'] for r in results),'source_images_labels_modified':False,'root_visual_review_pending':True,'model_predictions_tested':False,'training_accepted':False,'limitations':['Numerical and data-consumer checks do not prove trained model accuracy.','Original aid uses unchanged point/mask checks; Host source validation is consumed, not rerun.','Actual images are read separately for visual and Dataset consumers; no claim of single disk read.']};write_new(output/'REPORT.json',report);return {k:report[k] for k in ('status','count','instance_count','workers','elapsed_seconds','max_native_round_trip_error_px')}
 except BaseException as e:
  write_new(output/'FAILED.json',{'error':repr(e),'traceback':traceback.format_exc(),'completed_sample_count':len(results),'training_accepted':False});raise

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--data-root',required=True);p.add_argument('--pattern',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=128);p.add_argument('--workers',type=int,default=36);p.add_argument('--threads',type=int,default=2);p.add_argument('--cpu-budget',type=int,default=72);p.add_argument('--config',default='configs/model/action4_pilot2000_v002.json');a=p.parse_args(argv)
 print(json.dumps(run({'path':a.plan,'sha256':a.plan_sha},a.data_root,a.pattern,a.out,a.count,a.workers,a.threads,a.cpu_budget,a.config),indent=2));return 0
