import sys,json,argparse,os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
R=Path('/data/zy/vln/a-test');O=R/'outputs/action4_roi_adaptation_candidate_v001';sys.path[:0]=[str(O/'overlay'),str(R/'src')]
def main():
 import torch
 from model.training import action4_trainer as trainer
 from model.training.action4_runtime import input_path,sha256_file,canonical_hash
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();job=json.loads(input_path(R,a.config).read_text());cfg=json.loads(input_path(R,job['model_config']).read_text());out=R/a.output;assert out.resolve().is_relative_to(R/'outputs');out.mkdir(exist_ok=False)
 original_identity=trainer.source_identity
 overrides={f'src/model/training/{name}':str((O/'overlay/model/training'/name).relative_to(R)) for name in ('action4_trainer.py','action4_initialization.py','action4_gate.py')}
 def identity(root):
  value=original_identity(root)
  for logical,path in overrides.items():value['files'][logical]=sha256_file(R/path)
  value['sha256']=canonical_hash(value['files']);value['candidate_source_overrides']=overrides;value['scope']='explicit candidate component, not production';return value
 trainer.source_identity=identity
 original_builder=trainer._roi_builder;counts={'visible_total':0,'hidden_total':0,'visible_supervised':0,'hidden_supervised':0,'training_rois':0}
 def builder(images,records,*args,**kwargs):
  rois,meta=original_builder(images,records,*args,**kwargs)
  if kwargs.get('jitter') is True and kwargs.get('selected_instances') is not None:
   flags=meta['valid'].detach().cpu().tolist();assert len(flags)==len(records)
   for row,gi,valid in zip(records,kwargs['selected_instances'],flags):
    counts['training_rois']+=1
    for visible,supported in zip(row['instances'][gi]['corner_visible'],valid):
     key='visible' if visible else 'hidden';counts[key+'_total']+=1;counts[key+'_supervised']+=int(supported)
  return rois,meta
 trainer._roi_builder=builder
 result=trainer.train(R,cfg,stage='A0',run_id='roi_migration_gpu_component_v001',init_weights=job['checkpoint'],component_smoke=True,max_updates=2)
 manifest=json.loads((R/'outputs/action4/roi_migration_gpu_component_v001/run_manifest.json').read_text());assert manifest['initialization']['mode']=='weights_only_A0_roi_migration_new_optimizer';assert result['global_step']==2 and result['nonfinite_updates']==0;assert counts['hidden_total']>0 and counts['hidden_supervised']==counts['hidden_total'];assert counts['visible_supervised']==counts['visible_total'];report={'status':'ACTUAL_WARMSTART_WIDE_ROI_GPU_COMPONENT_PASS','scope':'two actual A0 component updates, not formal model accuracy or promotable weights','global_step':result['global_step'],'nonfinite_updates':result['nonfinite_updates'],'supervision_counts':counts,'initialization':manifest['initialization'],'candidate_source_overrides':overrides,'full_result':result};(out/'REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='full_result'}))
if __name__=='__main__':main()
