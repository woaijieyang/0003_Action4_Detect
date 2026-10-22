"""Explicit historical-physical / live-execution bridge; stdlib only, no code execution.
Candidate helper. Archives are read as bytes only, never imported or evaluated.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,re
ROOT=Path('/data/zy/vln/a-test')
SCHEMA='action4.execution_context_bridge.v1'
FLAGS={'dataset_generation_allowed':False,'training_accepted':False,'actual_acceptance_transferred':False}
EXISTING_EXECUTION_PATHS={'src/data/action4_modular_pose_controls.py','src/data/action4_modular_pixels.py','src/data/action4_modular_synthesis_backend.py','src/data/action4_synthesis.py'}
NEW_EXECUTION_PATHS={'src/data/action4_mask_semantics.py','src/data/action4_pose_formal_adapter.py','src/data/action4_execution_context_bridge.py'}
FORMAL_FACTORY_SOURCES={'formal_composition': 'src/data/action4_formal_composition.py', 'formal_continuation': 'src/data/action4_formal_continuation.py', 'formal_environment': 'src/data/action4_formal_environment.py', 'formal_forearm_attribution': 'src/data/action4_formal_forearm_attribution.py', 'formal_lift': 'src/data/action4_formal_lift.py', 'formal_lift_composition': 'src/data/action4_formal_lift_composition.py', 'formal_table_composition': 'src/data/action4_formal_table_composition.py'}
NEW_EXECUTION_PATHS=NEW_EXECUTION_PATHS|set(FORMAL_FACTORY_SOURCES.values())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _equal(a,b):return digest(a)==digest(b)
def _unique(pairs):
 out={}
 for k,v in pairs:
  if k in out:raise ValueError('Duplicate JSON key')
  out[k]=v
 return out
def _json(data):return json.loads(data,object_pairs_hook=_unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def _ref_shape(r):
 if not isinstance(r,dict) or set(r)!={'path','sha256'} or not isinstance(r['path'],str) or Path(r['path']).is_absolute() or '..' in Path(r['path']).parts or not re.fullmatch('[0-9a-f]{64}',r.get('sha256','')):raise ValueError('Exact bounded relative path/SHA reference required')
 return r
def _key(r):_ref_shape(r);return r['path'],r['sha256']
def _remember(deps,r):
 old=deps.get(r['path'])
 if old is not None and old!=r['sha256']:raise ValueError('Conflicting actual dependency path: '+r['path'])
 deps[r['path']]=r['sha256']
def _pointer(root,pointer):
 if not pointer.startswith('/') or pointer=='/':raise ValueError('Explicit nonroot JSON pointer required')
 parts=[s.replace('~1','/').replace('~0','~') for s in pointer[1:].split('/')];node=root
 for s in parts[:-1]:node=node[int(s)] if isinstance(node,list) else node[s]
 return node,int(parts[-1]) if isinstance(node,list) else parts[-1]
def reference_only_clone(value,replacements,allowed_pairs):
 """Existing release clone law narrowed to exact whole-reference replacements.
 No scalar, arbitrary subtree, metadata deletion or inferred replacement is allowed.
 """
 out=deepcopy(value);seen=set();allowed={(_key(a),_key(b)) for a,b in allowed_pairs}
 for row in replacements:
  if set(row)!={'pointer','old_ref','new_ref'} or row['pointer'] in seen:raise ValueError('Unique exact wrapper reference replacements')
  seen.add(row['pointer']);a,b=_key(row['old_ref']),_key(row['new_ref'])
  if (a,b) not in allowed:raise ValueError('Wrapper replacement not an explicitly authorized code/wrapper edge')
  node,key=_pointer(out,row['pointer'])
  if not _equal(node[key],row['old_ref']):raise ValueError('Wrapper original reference differs')
  node[key]=deepcopy(row['new_ref'])
 return out
def append_formal_factory_sources(value,source_additions,allowed_refs):
 """Only the exact seven reviewed canonical refs may extend a wearing factory.
 This is NOT a generic JSON append API. Existing references/physical values stay.
 """
 required={'schema','base_kind','wearing_base_config','sources','dataset_generation_allowed','training_accepted'}
 if not isinstance(value,dict) or set(value)!=required or value.get('schema')!='action4.wearing_sampling_base.v1' or value.get('base_kind')!='palmar_wearing' or value.get('dataset_generation_allowed') is not False or value.get('training_accepted') is not False:raise ValueError('Only nonaccepted palmar wearing sampling factory may add formal consumers')
 if not isinstance(value['sources'],dict) or not isinstance(source_additions,dict) or set(source_additions)!=set(FORMAL_FACTORY_SOURCES):raise ValueError('All and only seven explicit formal source keys required')
 if set(value['sources'])&set(FORMAL_FACTORY_SOURCES):raise ValueError('Formal additions may not replace any existing source key')
 if not isinstance(allowed_refs,(list,tuple)) or len(allowed_refs)!=7:raise ValueError('Exact seven reviewed transition-addition refs required')
 allowed={}
 for reference in allowed_refs:
  _ref_shape(reference)
  if reference['path'] in allowed:raise ValueError('Duplicate reviewed formal addition')
  allowed[reference['path']]=reference['sha256']
 if set(allowed)!=set(FORMAL_FACTORY_SOURCES.values()):raise ValueError('Reviewed additions must be the seven canonical paths')
 out=deepcopy(value)
 for key,path in FORMAL_FACTORY_SOURCES.items():
  r=source_additions[key];_ref_shape(r)
  if r['path']!=path or r['sha256']!=allowed[path]:raise ValueError('Factory formal source differs from reviewed active addition')
  out['sources'][key]=deepcopy(r)
 restored=deepcopy(out)
 for key in FORMAL_FACTORY_SOURCES:del restored['sources'][key]
 if not _equal(restored,value):raise ValueError('Existing factory/physical fields changed')
 return out

def _is_historical_y25_contact(value):
 # This legacy document has no schema field. Its exact cache-result status is
 # the type discriminator; _contact_compatibility_view validates its full gate.
 return isinstance(value,dict) and value.get('status')=='PASS_CACHE_MOUNT_VS_Y25_NO_SURFACE_HITS'
def _contact_compatibility_view(old_ref,before,replacements):
 if before.get('status')!='PASS_CACHE_MOUNT_VS_Y25_NO_SURFACE_HITS' or before.get('checked_pairs')!=6 or before.get('unresolved_surface_pairs')!=0 or before.get('scene_loaded') is not False or before.get('pose_evaluations') not in (0,False):raise ValueError('Only unchanged historical Y25 cache contact result may be wrapped')
 if len(replacements)!=1 or replacements[0]['pointer']!='/sources/neutral_config' or before['sources']['neutral_config']!=replacements[0]['old_ref']:raise ValueError('Contact compatibility view can migrate only neutral_config reference')
 identity=before.get('identity_rebinding',{})
 if not isinstance(identity,dict) or 'execution_context_bridge_view' in identity:raise ValueError('Explicit original identity provenance required; repeated view wrapping needs review')
 original_result=identity.get('original_evidence',old_ref);_ref_shape(original_result)
 return {'schema':'action4.historical_contact_compatibility_view.v1','original_evidence_ref':deepcopy(old_ref),'original_physical_result_ref':deepcopy(original_result),'original_neutral_ref':deepcopy(replacements[0]['old_ref']),'execution_neutral_ref':deepcopy(replacements[0]['new_ref']),'not_reexecuted':True,'compatibility_view_only':True,'actual_acceptance_transferred':False}
def _attach_contact_view(predicted,view):
 out=deepcopy(predicted);out.setdefault('identity_rebinding',{})['execution_context_bridge_view']=deepcopy(view);return out
class ExecutionBridge:
 def __init__(self,bridge_ref,current_source,dependencies=None,*,root=ROOT):
  self.root=Path(root).resolve();self.dependencies=dependencies if dependencies is not None else {};self.origins=[];self.contact_views=[];self.bridge_ref=deepcopy(bridge_ref);self._archives={}
  b=self.live_read(bridge_ref);required={'schema','status','historical_source_ref','current_source_ref','historical_bank_config_ref','historical_bounds_ref','current_bounds_ref','transition_policy_ref','wrapper_ledger_ref',*FLAGS}
  if set(b)!=required or b['schema']!=SCHEMA or b['status']!='EXPLICIT_PHYSICAL_CACHE_REUSE_ONLY' or any(b[k] is not v for k,v in FLAGS.items()):raise ValueError('Explicit nonaccepting bridge schema')
  self.spec=b;old=self.live_read(b['historical_source_ref']);new=self.live_read(b['current_source_ref']);self.historical_source=deepcopy(old);self.current_source=deepcopy(new)
  if not _equal(new,current_source):raise ValueError('Bridge current context differs from live prepared context')
  required_context={'W_from_capture','K_native','sides','wearing_frames','baseline_environment','base_kind','absence_policy','capabilities','frozen_refs','scene_config_ref'}
  if set(old)!=set(new) or not required_context<=set(old):raise ValueError('Full exact old/new context fields required')
  if not _equal({k:v for k,v in old.items() if k not in ('frozen_refs','scene_config_ref')},{k:v for k,v in new.items() if k not in ('frozen_refs','scene_config_ref')}):raise ValueError('Physical context changed')
  oldconfig=self.live_read(b['historical_bank_config_ref']);self.historical_bank_config=deepcopy(oldconfig);self.current_consumer_ref=deepcopy(oldconfig['consumer_module_ref'])
  if oldconfig['source_context_ref']!=b['historical_source_ref'] or oldconfig['source_context_sha256']!=digest(old):raise ValueError('Historical bank context identity mismatch')
  policy=self.live_read(b['transition_policy_ref'])
  if set(policy)!={'schema','status','transitions','additions',*FLAGS} or policy['schema']!='action4.exact_execution_transitions.v1' or policy['status']!='EXACT_TRANSITIONS_FOR_CONTROLLED_REVIEW' or any(policy[k] is not v for k,v in FLAGS.items()):raise ValueError('Exact transition policy required')
  expected=deepcopy(old['frozen_refs']);pairs=[];known=set();source_refs={**old['frozen_refs'],oldconfig['consumer_module_ref']['path']:oldconfig['consumer_module_ref']['sha256']}
  for row in policy['transitions']:
   if set(row)!={'old_ref','new_ref','archive_ref','candidate_ref','review_ref','purpose'} or row['purpose'] not in ('mask_semantics','formal_validation','host_batch_validation','physical_evidence_bridge'):raise ValueError('Exact declared execution transition required')
   a,c=row['old_ref'],row['new_ref'];ka,kc=_key(a),_key(c)
   if a['path'] in known or a['path'] not in EXISTING_EXECUTION_PATHS or c['path']!=a['path'] or a['sha256']==c['sha256'] or source_refs.get(a['path'])!=a['sha256']:raise ValueError('Transition must bind an actual old execution module, once')
   known.add(a['path']);_ref_shape(row['archive_ref'])
   if row['archive_ref']['path']==a['path'] or row['archive_ref']['sha256']!=a['sha256']:raise ValueError('Immutable separate original-code archive required')
   self.live_bytes(row['archive_ref']);candidate=self.live_bytes(row['candidate_ref']);live=self.live_bytes(c);self.live_bytes(row['review_ref'])
   if candidate!=live:raise ValueError('Live new module differs from exact reviewed candidate bytes')
   self._archives[ka]=deepcopy(row['archive_ref']);pairs.append((a,c))
   if a==oldconfig['consumer_module_ref']:self.current_consumer_ref=deepcopy(c)
   if a['path'] in expected:
    del expected[a['path']]
    if c['path'] in expected and expected[c['path']]!=c['sha256']:raise ValueError('Execution transition collision')
    expected[c['path']]=c['sha256']
  for row in policy['additions']:
   if set(row)!={'new_ref','candidate_ref','review_ref','purpose'} or row['purpose'] not in ('mask_semantics','formal_validation','host_batch_validation','physical_evidence_bridge'):raise ValueError('Exact explicit execution addition')
   c=row['new_ref'];_ref_shape(c)
   if c['path'] in expected or c['path'] not in NEW_EXECUTION_PATHS:raise ValueError('Addition must be a new exact Python helper')
   if self.live_bytes(c)!=self.live_bytes(row['candidate_ref']):raise ValueError('Added module differs from candidate')
   self.live_bytes(row['review_ref']);expected[c['path']]=c['sha256']
  ledger=self.live_read(b['wrapper_ledger_ref'])
  if set(ledger)!={'schema','rows'} or ledger['schema']!='action4.reference_only_wrapper_ledger.v1':raise ValueError('Exact wrapper ledger required')
  seenwrappers=set()
  for row in ledger['rows']:
   if set(row) not in ({'old_ref','new_ref','replacements'},{'old_ref','new_ref','replacements','compatibility_view'},{'old_ref','new_ref','replacements','formal_factory_sources'}):raise ValueError('Exact wrapper row')
   a,c=row['old_ref'],row['new_ref'];_ref_shape(a);_ref_shape(c)
   if a['path'] in seenwrappers or a['path']==c['path'] or Path(a['path']).suffix!='.json' or Path(c['path']).suffix!='.json':raise ValueError('Versioned unique JSON wrapper required')
   seenwrappers.add(a['path']);before=self.live_read(a);after=self.live_read(c);predicted=reference_only_clone(before,row['replacements'],pairs)
   if 'formal_factory_sources' in row:
    approved=[item['new_ref'] for item in policy['additions'] if item['new_ref']['path'] in FORMAL_FACTORY_SOURCES.values()]
    predicted=append_formal_factory_sources(predicted,row['formal_factory_sources'],approved)
   if _is_historical_y25_contact(before) and 'compatibility_view' not in row:raise ValueError('Historical Y25 contact migration requires explicit compatibility view metadata')
   if 'compatibility_view' in row:
    view=_contact_compatibility_view(a,before,row['replacements'])
    if not _equal(view,row['compatibility_view']):raise ValueError('Contact compatibility view metadata or nonreexecution flags differ')
    self.live_bytes(view['original_physical_result_ref']);predicted=_attach_contact_view(predicted,view);self.contact_views.append({'new_wrapper_ref':deepcopy(c),**view})
   if not _equal(after,predicted):raise ValueError('Wrapper changed beyond exact reference pointers and controlled contact-view metadata')
   pairs.append((a,c))
   if a['path'] in expected:
    del expected[a['path']]
    if c['path'] in expected:raise ValueError('Wrapper closure collision')
    expected[c['path']]=c['sha256']
  if not _equal(expected,new['frozen_refs']):raise ValueError('Unlisted execution closure change/addition/removal')
  if old['scene_config_ref']!=new['scene_config_ref'] and (_key(old['scene_config_ref']),_key(new['scene_config_ref'])) not in {(_key(a),_key(c)) for a,c in pairs}:raise ValueError('Scene factory must be an explicit reference-only wrapper')
  for p,h in new['frozen_refs'].items():self.live_bytes({'path':p,'sha256':h})
  oldbounds=self.live_read(b['historical_bounds_ref']);newbounds=self.live_read(b['current_bounds_ref'])
  if oldconfig['parameter_bounds_ref']!=b['historical_bounds_ref']:raise ValueError('Historical bounds not bound to old config')
  # Bounds reference migration itself must be a reference-only wrapper, preserving fixed/canonical/size/bounds semantics.
  if b['historical_bounds_ref']!=b['current_bounds_ref'] and (_key(b['historical_bounds_ref']),_key(b['current_bounds_ref'])) not in {(_key(a),_key(c)) for a,c in pairs}:raise ValueError('Bounds require exact wrapper migration')
  if not _equal({k:v for k,v in oldbounds.items() if k!='source_refs'},{k:v for k,v in newbounds.items() if k!='source_refs'}):raise ValueError('Canonical8, bounds, dimensions or policy changed')
  self.close()
 def _path(self,r):
  _ref_shape(r);p=(self.root/r['path']).resolve()
  if not p.is_relative_to(self.root):raise ValueError('Reference escapes project')
  return p
 def live_bytes(self,r):
  data=self._path(r).read_bytes()
  if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Live SHA mismatch: '+r['path'])
  _remember(self.dependencies,r);return data
 def live_read(self,r):return _json(self.live_bytes(r))
 def bytes(self,r,dependencies=None):
  actual=self._archives.get(_key(r),r);data=self.live_bytes(actual)
  if dependencies is not None:_remember(dependencies,actual)
  origin={'original_ref':deepcopy(r),'actual_read_ref':deepcopy(actual),'archived':actual!=r}
  if origin not in self.origins:self.origins.append(origin)
  return data
 def read(self,r,dependencies=None):return _json(self.bytes(r,dependencies))
 def close(self):
  for p,h in list(self.dependencies.items()):self.live_bytes({'path':p,'sha256':h})
  return {'bridge_ref':deepcopy(self.bridge_ref),'actual_dependency_refs':deepcopy(self.dependencies),'historical_read_origins':deepcopy(self.origins),'historical_contact_compatibility_views':deepcopy(self.contact_views),'physical_cache_reuse_only':True,**FLAGS}
 def assert_parameter_rebinding(self,historical_ref,current,current_config_ref,*,scene_contract=False):
  """Caller must reconstruct current with existing typed validator first.
  This checker then proves exact old/new tree equality except deterministic identity fields.
  """
  old=self.read(historical_ref)
  if scene_contract:
   if old.get('schema')!='action4.modular_dynamic_scene.v1' or old.get('source_context_sha256')!=digest(self.historical_source) or old.get('parameters_sha256')!=digest(old.get('parameters')):raise ValueError('Exact historical scene contract/source/parameters required')
   old=old['parameters']
  c=self.live_read(current_config_ref)
  if c.get('schema') not in ('action4.modular_pose_bank_config.v4','action4.modular_pose_bank_config.v5') or c.get('bridge_ref')!=self.bridge_ref or c['source_context_ref']!=self.spec['current_source_ref'] or c['parameter_bounds_ref']!=self.spec['current_bounds_ref']:raise ValueError('Current v4 config does not bind this bridge')
  if c['source_context_sha256']!=digest(self.current_source) or c['factory_ref']!=self.current_source['scene_config_ref'] or c['consumer_module_ref']!=self.current_consumer_ref:raise ValueError('Current config execution identity differs')
  self.live_bytes(c['consumer_module_ref'])
  prov=old['provenance']
  if prov['source_context_sha256']!=digest(self.historical_source) or prov['bank_config_ref']!=self.spec['historical_bank_config_ref'] or prov['pose_controls_module_ref']!=self.historical_bank_config['consumer_module_ref']:raise ValueError('Historical parameter provenance differs')
  expected=deepcopy(old);expected['provenance']['source_context_sha256']=digest(self.current_source);expected['provenance']['bank_config_ref']=deepcopy(current_config_ref);expected['provenance']['pose_controls_module_ref']=deepcopy(c['consumer_module_ref']);expected['provenance']['execution_context_bridge_ref']=deepcopy(self.bridge_ref)
  if not _equal(expected,current):raise ValueError('Parameter difference exceeds exact bound identity metadata; derived fields cannot be exempted')
  return {'historical_parameters_ref':deepcopy(historical_ref),'current_parameters_digest':digest(current),'bridge_ref':deepcopy(self.bridge_ref),'all_physical_and_derived_fields_exactly_preserved':True,**FLAGS}
def migrate_reference_wrapper(old_ref,replacements,allowed_pairs,output_path,*,root=ROOT,contact_compatibility=False,formal_factory_sources=None,allowed_factory_refs=()):
 """Reusable versioned clone, adapted from existing rebind_sources clone+ledger law.
 Pure reference migration only; exclusive create, no overwrite, no source mutation.
 """
 root=Path(root).resolve();_ref_shape(old_ref);oldpath=(root/old_ref['path']).resolve();out=(root/output_path).resolve()
 if not oldpath.is_relative_to(root) or not out.is_relative_to(root) or out==oldpath:raise ValueError('Bound separate output required')
 data=oldpath.read_bytes()
 if hashlib.sha256(data).hexdigest()!=old_ref['sha256']:raise ValueError('Old wrapper identity changed')
 before=_json(data)
 if _is_historical_y25_contact(before) and contact_compatibility is not True:raise ValueError('Historical Y25 contact migration cannot omit or disable compatibility view')
 value=reference_only_clone(before,replacements,allowed_pairs);view=None
 if formal_factory_sources is not None:
  if contact_compatibility:raise ValueError('Contact compatibility and formal factory append are distinct')
  value=append_formal_factory_sources(value,formal_factory_sources,allowed_factory_refs)
 if contact_compatibility:
  view=_contact_compatibility_view(old_ref,before,replacements);value=_attach_contact_view(value,view)
 for row in replacements:
  for r in (row['new_ref'],):
   p=(root/r['path']).resolve()
   if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('New reference identity mismatch')
 if formal_factory_sources is not None:
  for r in formal_factory_sources.values():
   p=(root/r['path']).resolve()
   if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Formal factory source addition identity mismatch')
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
 row={'old_ref':deepcopy(old_ref),'new_ref':{'path':str(out.relative_to(root)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},'replacements':deepcopy(replacements)}
 if view is not None:row['compatibility_view']=view
 if formal_factory_sources is not None:row['formal_factory_sources']=deepcopy(formal_factory_sources)
 return row
