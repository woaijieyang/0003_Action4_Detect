"""Left34 preparation only. Reuses unchanged right helper geometry/materials; no build."""
import numpy as np
from data import action4_brace_module as base
CONFIG='configs/data/action4_left_brace_module_v001.json'
def load_config(path=CONFIG):
 c=base.load(path)
 if c.get('schema')!='action4.left_brace_preparation.v1' or c.get('status')!='PREPARED_NOT_BUILT' or c.get('dataset_generation_allowed') is not False or c.get('build_authorized') is not False:raise ValueError('Preparation only')
 if c.get('side')!='left' or c.get('width_m')!=.034 or c.get('proximal_offset_from_wrist_m')!=.032 or c.get('segments')!=72:raise ValueError('Fixed left34 parameters')
 for ref in c['sources'].values():
  if set(ref)!={'path','sha256'} or base.sha(ref['path'])!=ref['sha256']:raise ValueError('Source identity mismatch')
 left=base.load(c['sources']['left_fit_config']['path'])
 if left['width_m']!=.034 or left['source_geometry_reference_checks']['new_center_proximal_from_wrist_m']!=.032:raise ValueError('Left fit reference mismatch')
 return c

def expected(c):
 # Existing parameters() supports side selection and checks analytic radii/reference.
 p=base.parameters(c)
 V,F,M,roles,rims=base.annular_geometry(p['inner'],p['thickness'],c['width_m'],c['segments'],p['trim_width'])
 return p,V,F,M,roles,rims

def verify_geometry(c,V,F,M):
 p,ref,faces,mats,roles,rims=expected(c);V=np.asarray(V);F=np.asarray(F);M=np.asarray(M)
 if V.shape!=ref.shape or not np.isfinite(V).all() or abs(V-ref).max()>1e-8 or not np.array_equal(F,faces) or not np.array_equal(M,mats):raise ValueError('Left source dimensions/topology/material assignment differ')
 e=np.concatenate([F[:,[i,(i+1)%4]] for i in range(4)]);u,inv,count=np.unique(np.sort(e,axis=1),axis=0,return_inverse=True,return_counts=True);balance=np.bincount(inv,weights=np.where(e[:,0]<e[:,1],1,-1))
 if not np.all(count==2) or not np.all(balance==0):raise ValueError('Closed shell winding')
 T=np.concatenate([F[:,[0,1,2]],F[:,[0,2,3]]]);volume=float(np.einsum('ij,ij->i',V[T[:,0]],np.cross(V[T[:,1]],V[T[:,2]])).sum()/6)
 expected_volume=c['width_m']*c['segments']/2*np.sin(2*np.pi/c['segments'])*(np.prod(p['outer'])-np.prod(p['inner']))
 if volume<=0 or abs(volume-expected_volume)>1e-11:raise ValueError('Shell volume')
 analytic=base.load(c['sources']['analytic34_reference']['path']);errors={}
 for name,ids in rims.items():
  error=float(abs(V[np.r_[ids,ids[0]]]-analytic['rings34_in_newreference'][name]).max());errors[name]=error
  if error>1e-8:raise ValueError('Left pointwise ring mismatch')
 return {'status':'PASS_PURE_LEFT34_GEOMETRY_NOT_BUILT','vertices':len(V),'quads':len(F),'triangles':len(T),'inner_radii_m':p['inner'].tolist(),'outer_radii_m':p['outer'].tolist(),'width_m':c['width_m'],'reference_m':.032,'pointwise_ring_errors_m':errors,'closed_winding':True,'volume_m3':volume,'target_geometry':False,'canonical8':None,'actual_asset_built':False},T

def validate_materials(c,band,contract=None):
 # Existing supported gray-weave/green nodes, packed source checks when built later.
 return base.material_states(c,band,contract)

def build(*args,**kwargs):
 raise RuntimeError('No actual build authorized by this prepared module; independent review required')
