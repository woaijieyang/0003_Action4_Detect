"""Applied backend CPU contracts. Every generated file is a labeled mock under outputs.
Planner allocation is tested in memory only; no Blender, GPU, real reviewed release or training execution.
"""
import ast
from copy import deepcopy
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/data/action4_synthesis.py').is_file());sys.path.insert(0,str(ROOT/'src'))
from data import action4_synthesis as synth
from data import action4_modular_synthesis_backend as backend
from data import action4_modular_scene_sampling as sampling
from data.action4_adapter import DataAdapter
BASELINE=ROOT/'outputs/action4_modular_batch_backend_prepare_v002'
OUT=ROOT/'outputs/action4_active_backend_fixture_fix_v001'
CRITERIA=('anatomy_and_pose','mount_and_cable_attachment','cloth_contact','lighting_and_material','native_roi_detail','collision_and_masks')

class FlowFixture:
 def __init__(self,count=8):
  self.dir=OUT/'CPU_MOCK_fixtures'/uuid.uuid4().hex;self.dir.mkdir(parents=True)
  self.original=json.loads((ROOT/'data/action4_synth/pilot2000_v015_candidate/plan.json').read_text())
  self.entries=deepcopy(self.original['entries'][:count])
  # Clearly named CPU fixture bytes, never a real geometry asset or review.
  geom=self.save('MOCK_NOT_A_GEOMETRY_CACHE.npz',{'fixture_only':True});asset=self.save('MOCK_NOT_A_BLENDER_ASSET.blend',{'fixture_only':True});canonical=self.save('MOCK_CANONICAL.json',{'fixture_only':True})
  self.config=self.save('sampling.json',{'schema':'action4.wearing_sampling_base.v1','base_kind':'palmar_wearing','fixture_only':True})
  self.bounds=self.save('bounds.json',{'fixture_only':True,'status':'MOCK_ENGINEERING_BOUNDS'})
  self.source={'base_kind':'palmar_wearing','scene_config_ref':self.config,'frozen_refs':{r['path']:r['sha256'] for r in [geom,asset,canonical]}}
  source=self.save('source_context.json',self.source);self.params={};self.rows={}
  for e in self.entries:
   p={'schema':'action4.modular_parameters.v1','entry':deepcopy(e),'estimated':True,'training_accepted':False,'dataset_generation_allowed':False,'augmentation':{'counts_as_formal_coverage':False},'provenance':{'source_context_sha256':backend._digest(self.source)},'fixture_only':True}
   self.params[e['sample_id']]=p;self.rows[e['sample_id']]={'entry':deepcopy(e),'parameters_ref':self.save('parameter_'+e['sample_id']+'.json',p)}
  release=self.save('parameter_release.json',{'schema':'action4.modular_parameter_release.v2','base_kind':'palmar_wearing','source_context_sha256':backend._digest(self.source),'sampling_bounds_ref':self.bounds,'entries':list(self.rows.values()),'fixture_only':True})
  self.recipe={'schema_version':synth.SCHEMA,'asset_backend':'modular_composed','families':deepcopy(self.original['recipe']['families']),'instance_weights':[.05,.15,.8],'resolution_wh':[3248,2464],'render':{'samples':64,'mask_samples':1,'device':'CPU','amodal_mode':'full'},'quality_gate':{'minimum_reviewed_candidates':128,'minimum_visible_fraction':.1,'minimum_visible_fullinput_area_px':16,'native_resolution_pending':False},'modular_backend':{'schema':'action4.modular_synthesis_backend.v2','base_kind':'palmar_wearing','scene_config_ref':self.config,'source_context_ref':source,'sampling_bounds_ref':self.bounds,'parameters_release_ref':release}}
  self.recipe_ref=self.save('recipe.json',self.recipe)
  self.context={'base_kind':'palmar_wearing','source_context':self.source,'source_context_sha256':backend._digest(self.source),'wearing_context':{'canonical_definition':{'source_manifest':canonical}}}
  # Real on-disk authority consumed by the unmocked opening index. Allocation
  # is the explicit CPU_MOCK leaf; separate tests below still exercise the real
  # 2000-entry planner and original0/1/2 quotas without this leaf patch.
  self.authority={'schema':'action4.modular_entry_authority.v1','status':'FROZEN_ENTRY_IDENTITIES_ONLY_NOT_RENDER_PLAN','entries':deepcopy(self.entries),'training_accepted':False,'dataset_generation_allowed':False,'fixture_only':'CPU_MOCK_NOT_PRODUCTION_AUTHORITY'}
  authority_ref=self.save('CPU_MOCK_ENTRY_AUTHORITY.json',self.authority)
  self.plan={'schema_version':synth.SCHEMA,'status':'render_candidates_pending_acceptance','seed':self.original['seed'],'entry_authority_ref':authority_ref,'fixture_only':True,'count':count,'entries':self.entries,'recipe':self.recipe,'recipe_path':self.recipe_ref['path'],'recipe_sha256':self.recipe_ref['sha256'],'synthesis_code_sha256':synth._hash(synth.__file__),'render_base_sha256':synth._hash('src/data/action4_render.py'),'asset_identities':backend.asset_identities(self.recipe),'split_counts':{s:sum(e['split']==s for e in self.entries) for s in ('train','eval','test')}}
  self.plan_ref=self.save('MOCK_PLAN_NOT_A_GENERATION_PLAN.json',self.plan)
  rgb=self.save('MOCK_IMAGE_BYTES_NOT_RENDER.png',{'fixture_only':True})
  self.records=[]
  for e in self.entries:
   rows=[{'instance_id':i,'object_id':1,'corners_px':[[10.,20.],[20.,20.],[20.,30.],[10.,30.]]*2,'corner_visible':[True]*8,'bbox_full_xyxy':[10.,20.,20.,30.],'mask_visible_sha256':'a'*64,'mask_amodal_expanded_sha256':'b'*64} for i in synth.target_indices(e) if i]
   self.records.append({**deepcopy(e),'fixture_only':True,'schema_version':'action4.v1','image':rgb['path'],'rgb_sha256':rgb['sha256'],'width':3248,'height':2464,'instances':rows,'excluded_instances':[],'planned_target_count':e['target_count'],'asset_backend':'modular_composed','training_accepted':False,'dataset_generation_allowed':False,'status':'render_candidate_not_training_accepted','synthesis_code_sha256':self.plan['synthesis_code_sha256'],'render_base_sha256':self.plan['render_base_sha256'],'pixel_source_identities':deepcopy(self.plan['asset_identities']),'modular_backend':{'base_kind':'palmar_wearing','entry_sha256':backend._digest(e),'plan_sha256':self.plan_ref['sha256'],'recipe_sha256':self.recipe_ref['sha256'],'parameters_ref':self.rows[e['sample_id']]['parameters_ref'],'parameters_sha256':backend._digest(self.params[e['sample_id']]),'source_context_sha256':self.context['source_context_sha256'],'parameter_counts_as_formal_coverage':False}})
  affine,_=DataAdapter.letterbox_affine((3248,2464),(1024,768))
  for record in self.records:
   for instance in record['instances']:
    instance.update(ignore=False,area_visible_px=1000,area_inframe_amodal_px=2000,area_full_amodal_px=2000,visible_area_at_fullinput_px=1000*float(affine[0,0]*affine[1,1]),bbox_visible_xyxy=[10.,20.,40.,60.])
  self.shard=self.dir/'shard';self.shard.mkdir();self.runtime=self.dir/'runtime';self.runtime.mkdir();self.sync_records();(self.runtime/'launch.json').write_text(json.dumps({'plan_sha256':self.plan_ref['sha256'],'command':['CPU_MOCK_NO_RENDER','--plan',self.plan_ref['path']],'expected_sample_ids':[e['sample_id'] for e in self.entries]}))
  self.bindings=backend.generation_bindings(self.plan,self.plan_ref,self.context)
  # Review objects exist only in mocked read results, not as actual approved files.
  self.spec=self.save('MOCK_SPEC_NO_ACTUAL_REVIEW.json',{'fixture_only':True,'not_a_review':True});self.root=self.save('MOCK_ROOT_NO_ACTUAL_REVIEW.json',{'fixture_only':True,'not_a_review':True});self.generation=self.save('MOCK_RELEASE_NO_ACTUAL_APPROVAL.json',{'fixture_only':True,'not_a_release':True})
  self.mock_docs={self.spec['path']:{'schema':'action4.modular_generation_review.v1','role':'spec','decision':'approved_generation','reviewer':'CPU_MOCK_ONLY','observations':'Synthetic branch test, never an actual review','binding_sha256':backend._digest(self.bindings)},self.root['path']:{'schema':'action4.modular_generation_review.v1','role':'root','decision':'approved_generation','reviewer':'CPU_MOCK_ONLY','observations':'Synthetic branch test, never an actual review','binding_sha256':backend._digest(self.bindings)},self.generation['path']:{'schema':'action4.modular_generation_release.v1','status':'REVIEWED_FOR_BOUND_PLAN','bindings':self.bindings,'reviews':{'spec':self.spec,'root':self.root},'presence_quota_policy_ref':backend.CURRENT_PRESENCE_QUOTA_POLICY_REF}}
  self.acceptance={'fixture_only':True,'decision':'accepted','plan_sha256':self.plan_ref['sha256'],'criteria':{k:True for k in CRITERIA},'reviewed_samples':[{'sample_id':r['sample_id'],'reviewer':'CPU_FIXTURE_ONLY','observations':'MOCK, not an image review','decision':'accepted','rgb_sha256':r['rgb_sha256']} for r in self.records],'generation_release_ref':self.generation}
  self.real_read=synth._read
 def save(self,name,value):
  p=self.dir/name;p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return {'path':str(p.relative_to(ROOT)),'sha256':synth._hash(p)}
 def read_mock(self,path):
  key=str(synth._path(path).relative_to(ROOT));return deepcopy(self.mock_docs[key]) if key in self.mock_docs else self.real_read(path)
 def sync_records(self):
  (self.shard/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))
 def assemble(self,acceptance=None,records=None):
  if records is not None:self.records=records;self.sync_records()
  aref=None if acceptance is None else self.save('acceptance_'+uuid.uuid4().hex+'.json',acceptance)
  dest=self.dir/('CPU_MOCK_ASSEMBLY_'+uuid.uuid4().hex)
  with self.mock_environment():
   result=synth.assemble(self.plan_ref['path'],[self.shard],dest,aref['path'] if aref else None)
  return result,dest

 def validate_cpu_mock_records(self,directory,records=None,validation_session=None):
  # Only image/geometry leaves are mocked here. The public validate_shard,
  # real HostValidationSession/index/source hashes/record checks and deferred
  # publication remain active. PNG behavior has its separate actual-PNG suite.
  if validation_session is None:raise AssertionError('Real host session required')
  rows=records if records is not None else synth._jsonl(Path(directory)/'manifest.jsonl')
  for record in rows:backend.validate_host_record(record,validation_session=validation_session)
  report={'status':'CPU_MOCK_GEOMETRY_AND_PNG_LEAVES_NOT_ACTUAL_VALIDATION','fixture_only':True,'sample_ids':[r['sample_id'] for r in rows],'training_accepted':False}
  validation_session.defer_report(self.runtime/'geometry_validation.json',report)
  return report
 @contextmanager
 def mock_environment(self):
  from data import action4_modular_pixels as pixels
  with ExitStack() as stack:
   leaves=[patch.object(synth,'_read',side_effect=self.read_mock),
    patch.object(synth,'_run_output_dir',return_value=self.runtime),
    patch.object(synth,'_validate_shard_records',side_effect=self.validate_cpu_mock_records),
    patch.object(synth,'build_entry_authority',side_effect=lambda *a,**kw:deepcopy(self.authority)),
    patch.object(sampling,'prepare_base',return_value=self.context),
    patch.object(sampling,'validate_parameters',side_effect=lambda p,c:deepcopy(p)),
    patch.object(sampling,'verify_sources'),
    patch.object(pixels,'validate_host_pngs',return_value={'status':'CPU_MOCK_PNG_LEAF_NO_IMAGE_VALIDATION','files':[]}),
    patch.object(backend,'validate_presence_quota',return_value=backend.CURRENT_PRESENCE_QUOTA_POLICY_REF)]
   for leaf in leaves:stack.enter_context(leaf)
   yield

class ModularBackendContracts(unittest.TestCase):
 @contextmanager
 def assertBatchRejects(self,message):
  # The batch boundary preserves the original rejection as __cause__ and adds
  # mandatory failed-closing evidence. Do not accept an arbitrary exception.
  with self.assertRaisesRegex(RuntimeError,'Host closing failed') as caught:
   yield
  self.assertIsInstance(caught.exception.__cause__,ValueError)
  self.assertRegex(str(caught.exception.__cause__),message)
  self.assertIn('Poisoned batch cannot close successfully',str(caught.exception))

 def test_real_batch_source_drift_poison_blocks_report_publication(self):
  f=FlowFixture();asset=f.dir/'MOCK_NOT_A_GEOMETRY_CACHE.npz';before=asset.read_bytes();report=f.dir/'CPU_MOCK_MUST_NOT_PUBLISH.json'
  with f.mock_environment():
   session=backend.HostValidationSession(f.plan_ref);session.__enter__()
   session.validate_record(f.records[0]);session.defer_report(report,{'fixture_only':'CPU_MOCK','training_accepted':False})
   try:
    asset.write_bytes(before+b'CPU_MOCK_DRIFT')
    with self.assertRaisesRegex(ValueError,'Poisoned batch cannot close successfully'):session.close()
    with self.assertRaisesRegex(ValueError,'No successful published host receipt'):_=session.receipt
    self.assertFalse(report.exists())
   finally:asset.write_bytes(before)
 def test_applied_source_scope_and_original_quota_sharder(self):
  old=(BASELINE/'source_before/action4_synthesis.py').read_text();current=Path(synth.__file__).read_text()
  funcs=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
  a,b=funcs(old),funcs(current);self.assertEqual({k for k in a if a[k]!=b[k]},{'validate_recipe','_asset_identities','create_plan','validate_shard','assemble','_render_one','_render_inside_blender'})
  real=json.loads((ROOT/'data/action4_synth/pilot2000_v015_candidate/plan.json').read_text());self.assertEqual(real['split_counts'],{'train':1600,'eval':200,'test':200});self.assertEqual(sorted(e['index'] for i in range(4) for e in synth.select_shard_entries(real,i,4,entry_count=32)),list(range(128)))
  rows={e['sample_id']:{'entry':e} for e in real['entries']}
  with patch.object(backend,'_release',return_value=(None,None,None,rows)):backend.validate_plan(real)
  bad=deepcopy(real);e=next(e for e in bad['entries'] if e['target_count']==2);e['target_count']=1;e['target_presence']=[True,False];rows={e['sample_id']:{'entry':e} for e in bad['entries']}
  with patch.object(backend,'_release',return_value=(None,None,None,rows)):
   with self.assertRaisesRegex(ValueError,'presence quotas'):backend.validate_plan(bad)
  missing=deepcopy(real)
  for e in missing['entries']:
   if e['scene_family']=='cloth_lift':e['scene_family']='wall_rear'
  rows={e['sample_id']:{'entry':e} for e in missing['entries']}
  with patch.object(backend,'_release',return_value=(None,None,None,rows)):
   with self.assertRaisesRegex(ValueError,'all eight'):backend.validate_plan(missing)
 def test_new_test_dual_only_quota_uses_actual_planner_without_writing_plan(self):
  f=FlowFixture();recipe=deepcopy(f.recipe);recipe['instance_weights']=[.03,.05,.92];recipe['instance_weights_by_split']={'train':[.03,.05,.92],'eval':[.03,.05,.92],'test':[0.,0.,1.]};recipe_ref=f.save('MOCK_NEW92_RECIPE_NOT_A_PLAN.json',recipe);dest=f.dir/'MUST_NOT_WRITE_2000_PLAN.json'
  with patch.object(synth,'validate_recipe',return_value=recipe),patch.object(synth,'_asset_identities',return_value={}),patch.object(backend,'validate_plan',side_effect=lambda p:p),patch.object(synth,'_write') as writer:
   generated=synth.create_plan(recipe_ref['path'],dest,count=2000,seed=f.original['seed']);writer.assert_called_once()
  self.assertFalse(dest.exists())
  self.assertEqual(generated['split_counts'],{'train':1600,'eval':200,'test':200})
  self.assertEqual(generated['instance_counts_by_split'],{'train':{'0':48,'1':80,'2':1472},'eval':{'0':6,'1':10,'2':184},'test':{'0':0,'1':0,'2':200}})
  self.assertEqual({k:sum(e['target_count']==k for e in generated['entries']) for k in (0,1,2)},{0:54,1:90,2:1856})
  self.assertTrue(all(e['scene_family']=='hard_negative' for e in generated['entries'] if e['target_count']==0))
  groups={}
  for e in generated['entries']:self.assertEqual(groups.setdefault(e['group_id'],e['split']),e['split'])
  rows={e['sample_id']:{'entry':e} for e in generated['entries']}
  with patch.object(backend,'_release',return_value=(None,None,None,rows)):backend.validate_plan(generated)
  backend.validate_presence_quota(generated,backend.CURRENT_PRESENCE_QUOTA_POLICY_REF)
  with self.assertRaisesRegex(ValueError,'current presence quota'):backend.validate_presence_quota(f.original,backend.CURRENT_PRESENCE_QUOTA_POLICY_REF)
  wrong=deepcopy(backend.CURRENT_PRESENCE_QUOTA_POLICY_REF);wrong['sha256']='f'*64
  with self.assertRaisesRegex(ValueError,'current reviewed'):backend.validate_presence_quota(generated,wrong)
 def test_per_split_weights_accept_test_zeros_and_reject_invalid_controls(self):
  f=FlowFixture();recipe=deepcopy(f.recipe);recipe['instance_weights_by_split']={'test':[0,0,1]};synth.validate_recipe(recipe)
  for override in [{'other':[0,0,1]},{'test':[0,0,0]},{'test':[-1,1,1]},{'test':[0,0,float('nan')]},{'test':[False,0,1]}]:
   recipe['instance_weights_by_split']=override
   with self.assertRaises(ValueError):synth.validate_recipe(recipe)
 def test_historical_global_planner_entries_exact_and_test_marker_fallback(self):
  f=FlowFixture();original_spec=importlib.util.spec_from_file_location('historical_active_synthesis',ROOT/'src/data/action4_synthesis.py');old=importlib.util.module_from_spec(original_spec);original_spec.loader.exec_module(old)
  def generate(module,recipe,seed):
   rr=f.save('MOCK_MEMORY_ONLY_RECIPE_'+uuid.uuid4().hex+'.json',recipe);dest=f.dir/('MUST_NOT_WRITE_PLAN_'+uuid.uuid4().hex+'.json')
   with patch.object(module,'validate_recipe',return_value=recipe),patch.object(module,'_asset_identities',return_value={}),patch.object(backend,'validate_plan',side_effect=lambda p:p),patch.object(module,'_write') as writer:
    result=module.create_plan(rr['path'],dest,count=2000,seed=seed);writer.assert_called_once()
   self.assertFalse(dest.exists());return result
  historical=deepcopy(f.recipe)
  old_plan=generate(old,historical,f.original['seed']);staged_plan=generate(synth,historical,f.original['seed'])
  for key in ('entries','split_counts','instance_counts_by_split','intended_instance_quotas_by_split','family_counts'):self.assertEqual(old_plan[key],staged_plan[key])
  recipe=deepcopy(f.recipe);recipe['instance_weights']=[.03,.05,.92];recipe['instance_weights_by_split']={'test':[0,0,1]}
  proposed=generate(synth,recipe,0);negative=next(i for i,x in enumerate(recipe['families']) if x['name']=='hard_negative');marker=proposed['entries'][negative]
  self.assertEqual(marker['split'],'test');self.assertEqual(marker['target_count'],2);self.assertNotEqual(marker['scene_family'],'hard_negative')
  self.assertEqual({e['scene_family'] for e in proposed['entries']},backend.FAMILIES)
  self.assertTrue(all(e['target_presence']==[True,True] and e['target_count']==2 and e['scene_family']!='hard_negative' for e in proposed['entries'] if e['split']=='test'))
  backend.validate_presence_quota(proposed,backend.CURRENT_PRESENCE_QUOTA_POLICY_REF)
  for count in (0,1):
   bad=deepcopy(proposed);e=next(e for e in bad['entries'] if e['split']=='test');e['target_count']=count;e['target_presence']=[False,False] if count==0 else [True,False]
   with self.assertRaisesRegex(ValueError,'current presence quota'):backend.validate_presence_quota(bad,backend.CURRENT_PRESENCE_QUOTA_POLICY_REF)
 def test_recipe_and_context_never_fall_back_to_old_A(self):
  f=FlowFixture();synth.validate_recipe(f.recipe)
  with patch.object(sampling,'prepare_base',return_value={'base_kind':'bilateral','source_context':f.source}):
   with self.assertRaisesRegex(ValueError,'factory'):backend.prepare_context(f.recipe)
  bad=deepcopy(f.recipe);bad['modular_backend']['base_kind']='bilateral'
  with self.assertRaises(ValueError):backend.validate_recipe(bad)
  bad=deepcopy(f.recipe);bad['resolution_wh']=[812,616]
  with self.assertRaisesRegex(ValueError,'native'):backend.validate_recipe(bad)
 def test_complete_entry_sparse_ID_source_and_plan_rejections(self):
  f=FlowFixture(32)
  for r in f.records:backend.validate_record(r,plan_sha256=f.plan_ref['sha256'],plan=f.plan);DataAdapter(ROOT).validate_sample(r,check_image=False)
  r=deepcopy(next(r for r in f.records if r['target_count']));r['instances'][0]['instance_id']=999
  with self.assertRaisesRegex(ValueError,'Sparse'):backend.validate_record(r)
  r=deepcopy(f.records[0]);r['seed']+=1
  with self.assertRaises(ValueError):backend.validate_record(r,f.entries[0])
  with self.assertRaisesRegex(ValueError,'another'):backend.validate_record(f.records[0],plan_sha256='f'*64)
 def test_candidate_path_stays_false_and_real_review_not_manufactured(self):
  f=FlowFixture();result,dest=f.assemble();self.assertFalse(result['training_accepted']);self.assertIsNone(result['generation_release_ref'])
  rows=[json.loads(x) for p in dest.glob('*.jsonl') for x in p.read_text().splitlines()];self.assertTrue(all(r['training_accepted'] is False for r in rows))
  self.assertEqual(f.real_read(f.generation['path']),{'fixture_only':True,'not_a_release':True})
 def test_reachable_mock_promotion_keeps_parameters_false(self):
  f=FlowFixture(128);before={k:deepcopy(v) for k,v in f.params.items()};result,dest=f.assemble(f.acceptance);self.assertTrue(result['training_accepted']);self.assertEqual(result['generation_release_ref'],f.generation)
  records=[json.loads(x) for p in dest.glob('*.jsonl') for x in p.read_text().splitlines()];self.assertEqual(len(records),128)
  self.assertTrue(all(r['status']=='training_accepted' and r['training_accepted'] for r in records))
  self.assertTrue(all(r['modular_backend']['parameter_counts_as_formal_coverage'] is False for r in records));self.assertEqual(f.params,before)
  self.assertTrue(all(f.real_read(row['parameters_ref']['path'])['training_accepted'] is False for row in f.rows.values()))
 def test_missing_stale_or_incomplete_independent_release_rejected(self):
  f=FlowFixture(128);a=deepcopy(f.acceptance);a.pop('generation_release_ref')
  with self.assertBatchRejects('release required'):f.assemble(a)
  f.mock_docs[f.generation['path']]['bindings']['base_kind']='bilateral'
  with self.assertBatchRejects('Generation release differs'):f.assemble(f.acceptance)
  f.mock_docs[f.generation['path']]['bindings']['base_kind']='palmar_wearing';f.mock_docs[f.spec['path']]['binding_sha256']='f'*64
  with self.assertBatchRejects('spec review'):f.assemble(f.acceptance)
 def test_128_sixcriteria_and_every_record_checks_remain(self):
  f=FlowFixture(128);a=deepcopy(f.acceptance);a['reviewed_samples']=a['reviewed_samples'][:127]
  with self.assertBatchRejects('Insufficient'):f.assemble(a)
  for criterion in CRITERIA:
   a=deepcopy(f.acceptance);a['criteria'][criterion]=False
   with self.assertBatchRejects('criterion'):f.assemble(a)
  rs=deepcopy(f.records);rs[-1]['background_seed']+=1
  with self.assertBatchRejects('entry'):f.assemble(f.acceptance,rs)
 def test_pending_family_is_not_waived_by_generation_review(self):
  f=FlowFixture(128)
  def validator(p,c):
   if p['entry']['scene_family']=='self_forearm_occlusion':raise sampling.PendingFamilyError(['actual family effect pending'])
   return p
  with f.mock_environment(),patch.object(sampling,'validate_parameters',side_effect=validator):
   with self.assertRaises(sampling.PendingFamilyError):backend.validate_generation_release(f.plan,f.plan_ref,f.records,f.acceptance)
 def test_formal_ignored_support_guard_preserves_candidate_diagnostics(self):
  f=FlowFixture(32);entry=next(e for e in f.entries if e['target_count']==2);base=deepcopy(next(r for r in f.records if r['sample_id']==entry['sample_id']));gate=f.recipe['quality_gate']
  self.assertEqual(backend.validate_formal_targets(base,entry,gate),[1,2])
  for update in [{'ignore':True},{'unobservable':True},{'status':'unobservable'},
                 {'ignore':False,'area_visible_px':0,'area_inframe_amodal_px':100,'area_full_amodal_px':100,'visible_area_at_fullinput_px':1e9},
                 {'area_visible_px':1000,'area_inframe_amodal_px':20000,'area_full_amodal_px':20000},
                 {'area_visible_px':100,'area_inframe_amodal_px':200,'visible_area_at_fullinput_px':1e9},
                 {'visible_area_at_fullinput_px':1e9},{'bbox_visible_xyxy':None}]:
   r=deepcopy(base);r['instances'][0].update(update)
   backend.validate_record(r,entry,f.plan_ref['sha256'],f.plan)
   with self.assertRaises(ValueError):backend.validate_formal_targets(r,entry,gate)
  boundary=deepcopy(base);boundary['width']=1024;boundary['height']=768
  for instance in boundary['instances']:instance.update(area_visible_px=16,area_inframe_amodal_px=160,area_full_amodal_px=160,visible_area_at_fullinput_px=16.)
  self.assertEqual(backend.validate_formal_targets(boundary,entry,gate),[1,2])
  boundary['instances'][0]['area_visible_px']=15;boundary['instances'][0]['visible_area_at_fullinput_px']=15.
  with self.assertRaises(ValueError):backend.validate_formal_targets(boundary,entry,gate)
 def test_formal_promotion_rejects_one_ignored_planned_target(self):
  f=FlowFixture(128);r=next(r for r in f.records if r['target_count']==2 and r['split']=='test');r['instances'][0]['ignore']=True;f.sync_records()
  with self.assertBatchRejects('Ignored/unobservable'):f.assemble(f.acceptance)
 def test_candidate_cannot_self_promote_or_bypass_host_hook(self):
  f=FlowFixture();r=deepcopy(f.records[0]);p=deepcopy(f.params[r['sample_id']]);p['augmentation']['counts_as_formal_coverage']=True;r['modular_backend']['parameters_ref']=f.save('tampered_parameter.json',p);r['modular_backend']['parameters_sha256']=backend._digest(p)
  with self.assertRaisesRegex(ValueError,'self-promote'):backend.validate_record(r)
  with f.mock_environment(),patch.object(backend,'validate_host_record',side_effect=ValueError('host PNG sentinel')):
   with self.assertBatchRejects('host PNG sentinel'):synth.validate_shard(f.shard)

if __name__=='__main__':unittest.main(verbosity=2)
