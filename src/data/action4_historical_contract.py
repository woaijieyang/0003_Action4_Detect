"""Schema-scoped historical JSON representation loader; no Blender or scene IO.

Only shader graph edge rows under right_state_before_left regain the tuple
type emitted by action4_global_environment._nodes. No field/value is removed,
rounded or replaced. The existing original verifier remains authoritative.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path('/data/zy/vln/a-test')
sys.dont_write_bytecode=True
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from data.action4_global_environment import digest
SCHEMA='action4.modular_bilateral_scene.contract.v1'

def _pointer(value):return str(value).replace('~','~0').replace('/','~1')

def _tree(tree,path,converted):
 if tree is None:return
 if not isinstance(tree,dict) or set(tree)!={'nodes','links'} or not isinstance(tree['nodes'],list) or not isinstance(tree['links'],list):raise ValueError('Unsupported historical shader graph schema at '+path)
 for i,edge in enumerate(tree['links']):
  if not isinstance(edge,(list,tuple)) or len(edge)!=4 or type(edge[0]) is not str or not edge[0] or type(edge[1]) is not int or edge[1]<0 or type(edge[2]) is not str or not edge[2] or type(edge[3]) is not int or edge[3]<0:raise ValueError('Historical shader link must be (from-name, nonnegative socket-index, to-name, nonnegative socket-index)')
  if type(edge) is list:tree['links'][i]=tuple(edge);converted.append(path+'/links/'+str(i))

def normalize_historical_contract(value,*,return_receipt=False):
 if not isinstance(value,dict) or value.get('schema')!=SCHEMA:raise ValueError('Explicit supported bilateral historical contract schema required')
 before=digest(value);out=deepcopy(value);converted=[]
 try:
  right=out['right_state_before_left'];objects=right['objects'];world=right['common']['world']
  if not isinstance(objects,dict) or not isinstance(world,dict):raise ValueError('Malformed right source snapshot')
  for name,row in objects.items():
   path='/right_state_before_left/objects/'+_pointer(name)
   mesh,light=row['mesh'],row['light']
   if mesh is not None:
    materials=mesh['material_states']
    if not isinstance(materials,list):raise ValueError('Historical mesh material_states must be list')
    for i,material in enumerate(materials):_tree(material['nodes'],path+'/mesh/material_states/'+str(i)+'/nodes',converted)
   if light is not None:_tree(light['nodes'],path+'/light/nodes',converted)
  _tree(world['nodes'],'/right_state_before_left/common/world/nodes',converted)
 except (KeyError,TypeError) as exc:raise ValueError('Historical snapshot schema lacks explicit shader paths') from exc
 after=digest(out)
 if before!=after:raise ValueError('Normalization changed historical canonical fields or values')
 receipt={'schema':SCHEMA,'normalization':'only declared shader node edge rows list_to_tuple','canonical_digest_before':before,'canonical_digest_after':after,'canonical_digest_exactly_preserved':True,'converted_paths':converted,'all_other_fields_and_values_preserved':True}
 return (out,receipt) if return_receipt else out

def _unique_object(pairs):
 value={}
 for key,item in pairs:
  if key in value:raise ValueError('Duplicate historical JSON field '+key)
  value[key]=item
 return value

def load_historical_contract(reference,*,return_receipt=False):
 """Read a trusted relative path/SHA; return the original-value typed contract."""
 if not isinstance(reference,dict) or set(reference)!={'path','sha256'} or not isinstance(reference['path'],str) or Path(reference['path']).is_absolute() or '..' in Path(reference['path']).parts or not isinstance(reference['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',reference['sha256']):raise ValueError('Exact historical relative path/SHA256 reference required')
 path=(ROOT/reference['path']).resolve()
 if not path.is_relative_to(ROOT):raise ValueError('Historical source escapes project boundary')
 payload=path.read_bytes();before_sha=hashlib.sha256(payload).hexdigest()
 if before_sha!=reference['sha256']:raise ValueError('Historical source SHA256 differs')
 raw=json.loads(payload,object_pairs_hook=_unique_object)
 result,receipt=normalize_historical_contract(raw,return_receipt=True)
 after_sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if after_sha!=before_sha:raise ValueError('Historical source changed during loading')
 receipt.update({'historical_source_ref':deepcopy(reference),'historical_file_SHA256_before':before_sha,'historical_file_SHA256_after':after_sha,'historical_source_SHA256_exactly_preserved':True})
 return (result,receipt) if return_receipt else result
