"""Explicit phase/source isolation for canonical Action4 render and training jobs.
Historical files remain readable evidence, never implicit execution authority.
"""
from pathlib import Path
import hashlib,json
POINTER='configs/execution/action4_active.json'
def bounded(root,value):
 root=Path(root).resolve();p=Path(value)
 if '..' in p.parts:raise ValueError('Authority path traversal')
 p=(p if p.is_absolute() else root/p).resolve()
 if not p.is_relative_to(root):raise ValueError('Authority project boundary')
 return p
def ref(root,path):
 p=bounded(root,path);return {'path':str(p.relative_to(Path(root).resolve())),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def readref(root,r):
 p=bounded(root,r['path']);raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Execution authority reference changed: '+r['path'])
 return json.loads(raw)
def active(root):
 pointer_ref=ref(root,POINTER);pointer=readref(root,pointer_ref)
 if pointer.get('schema')!='action4.active_execution_pointer.v1':raise ValueError('Explicit active execution pointer required')
 rr=pointer['active_ref']
 if not bounded(root,rr['path']).is_relative_to(bounded(root,'configs/execution')):raise ValueError('Authority revision outside config directory')
 a=readref(root,rr)
 if a.get('schema')!='action4.execution_authority.v1':raise ValueError('Unknown execution authority')
 return a,[pointer_ref,rr]
def authorize_render(root,entrypoint,plan_path,output,shard_index=0,num_shards=1,entry_start=0,entry_count=None,limit=None):
 from .action4_synthesis import select_shard_entries
 a,refs=active(root);r=a.get('render',{})
 if a.get('status')!='ACTIVE' or r.get('enabled') is not True or a.get('phase') not in ('render128','render_remaining'):raise ValueError('Current execution authority does not permit rendering')
 if r.get('entrypoint') not in ('scripts/action4_render_cached.py','scripts/action4_render_fast.py') or str(bounded(root,entrypoint).relative_to(Path(root).resolve()))!=r['entrypoint']:raise ValueError('Legacy/unapproved render entrypoint')
 pr=ref(root,plan_path)
 if pr!=r['plan_ref']:raise ValueError('Old or unapproved render plan')
 plan=readref(root,pr);backend=plan['recipe']['modular_backend']
 if plan['count']!=2000 or backend['source_context_ref']!=a['source_context_ref'] or backend['parameters_release_ref']!=a['parameters_release_ref']:raise ValueError('Plan source/parameter version differs from active authority')
 chosen=select_shard_entries(plan,shard_index,num_shards,limit,entry_start,entry_count);lo,hi=r['index_range']
 if [lo,hi]!=([0,128] if a['phase']=='render128' else [128,2000]):raise ValueError('Phase range contract changed')
 if not chosen or any(not lo<=e['index']<hi for e in chosen):raise ValueError('Sample range crosses authorized execution phase')
 target=bounded(root,output);parent=bounded(root,r['output_parent'])
 if target.parent!=parent or not target.name.startswith(r['output_name_prefix']):raise ValueError('Output outside active phase namespace')
 if a['phase']=='render128':
  evidence=readref(root,r['geometry_ref'])
  if evidence.get('status')!='128_GEOMETRY_SELECTED_MASKS_PENDING' or evidence.get('selected')!=128:raise ValueError('Actual128 geometry evidence required')
  release=readref(root,a['parameters_release_ref'])
  if evidence.get('selected_rows')!=release['entries'][:128]:raise ValueError('Actual128 selected rows differ from active parameters')
  refs.extend([r['geometry_ref'],a['parameters_release_ref']])
 else:
  evidence=readref(root,r['accepted128_ref'])
  if evidence.get('status')!='ROOT_ACCEPTED_ACTUAL128_FOR_REMAINING1872' or evidence.get('plan_ref')!=pr or evidence.get('sample_ids')!=[e['sample_id'] for e in plan['entries'][:128]]:raise ValueError('Exact actual128 acceptance required before remaining images')
  refs.append(r['accepted128_ref'])
 if r['entrypoint']=='scripts/action4_render_fast.py':
  policy=readref(root,r['execution_policy_ref'])
  if policy.get('schema')!='action4.fast_execution_policy.v1' or policy.get('plan_ref')!=pr:raise ValueError('Fast execution policy mismatch')
  refs.extend([r['execution_policy_ref'],policy['handoff_ref']])
  readref(root,policy['handoff_ref'])
  for implementation in policy['implementation_refs']:
   if ref(root,implementation['path'])!=implementation:raise ValueError('Fast execution implementation drift')
  refs.extend(policy['implementation_refs'])
 refs.extend([pr,a['source_context_ref']]);readref(root,a['source_context_ref'])
 return refs

def flag(argv,name,default=None):
 token='--'+name;values=[]
 for i,a in enumerate(argv):
  if a==token:
   if i+1>=len(argv) or argv[i+1].startswith('--'):raise ValueError('Missing authority argument value')
   values.append(argv[i+1])
  elif a.startswith(token+'='):values.append(a.split('=',1)[1])
 if len(values)>1:raise ValueError('Duplicate authority argument')
 return values[0] if values else default

def template_training_references(root,t,dataset,config,plan):
 """Admit the explicit committed template pilot; trainer verifies full release."""
 splits={'train':1600,'eval':200,'test':200}
 layout={s:'data/'+s for s in splits};layout['commit']='data/action4_dataset_commit.json'
 if (plan.get('schema')!='action4.fixed_template_batch_plan.v1' or plan.get('count')!=2000 or len(plan.get('entries',[]))!=2000 or plan.get('split_counts')!=splits):raise ValueError('Current fixed-template2000 plan required')
 if (dataset.get('schema')!='action4.template_dataset_commit.v1' or dataset.get('status')!='COMPLETE' or dataset.get('complete') is not True or dataset.get('training_accepted') is not True or dataset.get('count')!=2000 or dataset.get('split_counts')!=splits or dataset.get('output_layout')!=layout):raise ValueError('Complete current fixed-template dataset commit required')
 if t['dataset_manifest_ref']['path']!=layout['commit'] or dataset.get('plan_ref')!=t['plan_ref'] or config['data'].get('dataset_commit_ref')!=t['dataset_manifest_ref']:raise ValueError('Dataset commit/config/plan binding differs')
 rr=t['release_ref'];pr=t['split_protocol_ref'];release=readref(root,rr);protocol=readref(root,pr)
 if (release.get('schema')!='action4.template_generation_release.v1' or release.get('status')!='FORMAL_2000_RELEASE_APPROVED' or set(release.get('reviews',{}))!={'root','spec'}):raise ValueError('Formal template release required')
 binding=release['binding']
 if (binding.get('plan_ref')!=t['plan_ref'] or binding.get('output_layout')!=layout or binding.get('split_protocol_ref')!=pr or dataset.get('release_ref')!=rr or dataset.get('split_protocol_ref')!=pr or dataset.get('ledger_ref')!=binding.get('ledger_ref') or dataset.get('relocation_ledger_ref')!=binding.get('relocation_ledger_ref')):raise ValueError('Release/commit binding differs')
 if (config['data'].get('dataset_release_ref')!=rr or config['data'].get('split_protocol_ref')!=pr or protocol.get('plan_ref')!=t['plan_ref'] or protocol.get('mode')!='independent_samples_shared_scene_templates' or protocol.get('output_layout')!=layout):raise ValueError('Explicit shared-template protocol required')
 if set(t['split_refs'])!=set(splits) or dataset.get('manifest_refs')!=t['split_refs']:raise ValueError('Committed train/eval/test references required')
 refs=[rr,pr]
 binding_sha=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 for role,review_ref in release['reviews'].items():
  review=readref(root,review_ref)
  if review.get('role')!=role or review.get('decision')!='approve_formal_2000_release' or review.get('binding_sha256')!=binding_sha:raise ValueError('Bound formal review required')
  refs.append(review_ref)
 for split,r in t['split_refs'].items():
  if r['path']!=layout[split]+'/annotations.jsonl' or config['data'][split+'_manifest']!=r['path'] or ref(root,r['path'])!=r:raise ValueError('Old or mixed training split')
  refs.append(r)
 return refs

def spec_references(root,spec):
 argv=spec.get('argv',[])
 if not isinstance(argv,list) or len(argv)<2:raise ValueError('Explicit execution argv required')
 if spec.get('kind')=='render':
  if argv[2:3]!=['render-shard']:raise ValueError('Public render-shard command required')
  return authorize_render(root,argv[1],flag(argv,'plan'),flag(argv,'output'),int(flag(argv,'shard-index','0')),int(flag(argv,'num-shards','1')),int(flag(argv,'entry-start','0')),int(flag(argv,'entry-count')) if flag(argv,'entry-count') is not None else None,int(flag(argv,'limit')) if flag(argv,'limit') is not None else None)
 if spec.get('kind')=='quality':
  a,refs=active(root);q=a.get('quality',{})
  if a.get('status')!='ACTIVE' or a.get('phase') not in ('render_remaining','quality') or q.get('enabled') is not True:raise ValueError('Quality phase not authorized')
  if str(bounded(root,argv[1]).relative_to(Path(root).resolve()))!='scripts/action4_quality.py':raise ValueError('Unapproved quality entry')
  accepted=readref(root,q['accepted128_ref'])
  if accepted.get('status')!='ROOT_ACCEPTED_ACTUAL128_FOR_REMAINING1872' or accepted.get('plan_ref')!=q['plan_ref']:raise ValueError('Actual128 root acceptance required before GPU3')
  cr=ref(root,flag(argv,'config'))
  if cr not in q['config_refs']:raise ValueError('Unapproved quality config')
  config=readref(root,cr)
  if config.get('plan_ref')!=q['plan_ref'] or config.get('gpu_uuid')!='GPU-510dc520-5a9d-1317-019d-de5ed02c1b37':raise ValueError('Quality config scope mismatch')
  return refs+[q['accepted128_ref'],q['plan_ref'],cr]
 if spec.get('kind')=='train':
  a,refs=active(root);t=a.get('training',{})
  if a.get('status')!='ACTIVE' or a.get('phase')!='training' or t.get('enabled') is not True:raise ValueError('Current execution authority does not permit training')
  if str(bounded(root,argv[1]).relative_to(Path(root).resolve()))!='scripts/action4_train.py' or flag(argv,'stage') not in t['stages']:raise ValueError('Unapproved training entry/stage')
  cr=ref(root,flag(argv,'config'))
  if cr!=t['config_ref'] or flag(argv,'local-config') is not None:raise ValueError('Training config differs from active authority')
  dataset=readref(root,t['dataset_manifest_ref']);config=readref(root,cr)
  plan=readref(root,t['plan_ref'])
  if t.get('dataset_protocol') is not None:
   if t['dataset_protocol']!='fixed_template_commit_v1':raise ValueError('Unknown explicit training dataset protocol')
   return refs+[cr,t['dataset_manifest_ref'],t['plan_ref']]+template_training_references(root,t,dataset,config,plan)
  if plan['recipe']['modular_backend']['source_context_ref']!=a['source_context_ref'] or plan['recipe']['modular_backend']['parameters_release_ref']!=a['parameters_release_ref']:raise ValueError('Training plan source differs from active authority')
  if set(t['split_refs'])!={'train','eval','test'} or dataset.get('split_counts')!={'train':1600,'eval':200,'test':200}:raise ValueError('Complete current train/eval/test bindings required')
  if dataset.get('training_accepted') is not True or dataset.get('count')!=2000 or dataset.get('plan_sha256')!=t['plan_ref']['sha256']:raise ValueError('Accepted current2000 dataset required')
  for split,r in t['split_refs'].items():
   if config['data'][split+'_manifest']!=r['path'] or ref(root,r['path'])!=r:raise ValueError('Old or mixed training split')
   refs.append(r)
  return refs+[cr,t['dataset_manifest_ref'],t['plan_ref']]
 return []
