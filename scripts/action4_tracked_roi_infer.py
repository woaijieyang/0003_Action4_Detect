"""Prediction-only A0 on authenticated completed tracking ROIs."""
import os,sys,json,copy,hashlib,traceback,time
from pathlib import Path
R=Path('/data/zy/vln/a-test')
def run():
 S=Path(__file__).resolve().parent;O=S.parent
 assert Path.cwd().resolve()==R
 sys.dont_write_bytecode=True;os.environ['ACTION4_PROJECT_ROOT']=str(R)
 sys.path[:]=[str(S/'runtime/surface_deps'),str(S/'src')]+[p for p in sys.path if p and not Path(p).resolve().is_relative_to(R/'src') and Path(p).resolve()!=R]
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--part',type=int,required=True);a=p.parse_args()
 cfg=json.loads((S/'CONFIG.json').read_text());job=cfg['parts'][a.part]
 import numpy as np,torch
 from eval.action4_reanchor_roi_provider import ReanchorROIProvider
 from eval.action4_staged_pipeline import StagedPipeline
 from eval.action4_sam_full_review import ReceiptVerifier
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 assert os.environ['CUDA_VISIBLE_DEVICES']==job['gpu_uuid']
 assert sorted(os.sched_getaffinity(0))==sorted(job['cpus'])
 torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 v=ReceiptVerifier()
 baseline=[json.loads(l) for l in v.verify(cfg['baseline']['path'],cfg['baseline']['sha256']).open()]
 assert len(baseline)==2850
 provider=ReanchorROIProvider(**cfg['provider'])
 for role in ['A0','B0']:v.verify(cfg['models'][role]['path'],cfg['models'][role]['sha256'])
 pipe=StagedPipeline.from_bundles(cfg['models']['A0']['path'],cfg['models']['B0']['path'])
 assert pipe.identity['A0_step']==15770 and pipe.identity['B0_step']==9500
 K=np.array(json.loads(v.verify(cfg['K']['path'],cfg['K']['sha256']).read_text())['K_native'])
 cad=np.array(json.loads(v.verify(cfg['CAD']['path'],cfg['CAD']['sha256']).read_text())['corners_3d_mm'])/1000
 seeds=json.loads(v.verify(cfg['seeds']['path'],cfg['seeds']['sha256']).read_text())['seeds']
 import re
 bootstrap={}
 for f in range(4):
  bootstrap[f]={}
  for seed in seeds:
   matches=[re.fullmatch(r'(?:b0:)?frame(\d+):grid:(\d+)',t) for t in seed['evidence_ids']]
   grid=[int(m[2]) for m in matches if m and int(m[1])==f]
   assert len(grid)==1
   bootstrap[f][grid[0]]=seed['track_id']
 out=O/f'part_{a.part:03d}';out.mkdir(exist_ok=False);calls={'B0':0,'A0':0}
 def h(name):
  def count(*args):calls[name]+=1
  return count
 hooks=[pipe.b0.register_forward_hook(h('B0')),pipe.a0.register_forward_hook(h('A0'))]
 results=[];start=time.time()
 try:
  for f in range(job['start'],job['end']):
   rgb,context=provider.load(f,K)
   raw=baseline[f];assert raw['frame_id']==f and raw['rgb_sha256']==provider.rows[f]['frame']['sha256']
   assert raw['prediction']['identity']['A0_step']==15770 and raw['prediction']['identity']['B0_step']==9500
   if f<4:
    prediction=copy.deepcopy(raw['prediction'])
    assert len(prediction['instances'])==2
    for instance in prediction['instances']:
     instance['track_id']=bootstrap[f][instance['grid_index']]
     instance['offline_confirmation_frame']=4
    scope='exact baseline A0 cache; offline track identity confirmed atframe4'
   else:
    prediction=pipe.predict_video(rgb,K,cad,video_context=context)
    for t in prediction['video_roi_consumption'].get('instances',[]):
     np.testing.assert_allclose(np.array(t['K112']),np.array(t['A112'])@K,rtol=2e-6,atol=2e-5)
    scope='current generation-bound tracking ROI; no SAM feature injection'
   row={'frame_id':f,'baseline':raw['prediction'],'fused':prediction,'input_scope':scope,
        'source_generation':provider.rows[f].get('generation'),'RGB_sha256':raw['rgb_sha256']}
   (out/f'frame_{f:08d}.json').write_text(json.dumps(row,allow_nan=False))
   results.append({'frame_id':f,'instances':len(prediction['instances']),'invalid':sum(not x['pose_valid'] for x in prediction['instances'])})
   if len(results)%25==0:(out/'STATUS.json').write_text(json.dumps({'frames':len(results),'expected':job['end']-job['start'],'seconds':time.time()-start}))
  assert calls['B0']==0 and calls['A0']==sum(r['instances'] for r in results if r['frame_id']>=4)
  (out/'COMPLETE.json').write_text(json.dumps({'status':'TRACKED_ROI_A0_PART_COMPLETE_NOT_ACCURACY','frames':results,'models':pipe.identity,'calls':calls,'seconds':time.time()-start},indent=2))
 except BaseException:
  (out/'FAILED.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
 finally:
  for hook in hooks:hook.remove()
if __name__=='__main__':run()
