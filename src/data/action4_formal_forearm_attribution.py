"""Actual evaluated-geometry right-forearm nearest-hit attribution.
A fixed 17x17 bbox ray grid is an attribution witness, never a pixel fraction.
"""
import numpy as np
from data import action4_global_environment as env
from data import action4_modular_assembly as assembly
from data import action4_modular_pixels as pixels
SEMANTIC_REF={'path': 'data/action4_scene_assets/modular_wrist_v001/build_v003/right_geometry.npz', 'sha256': 'f282c4d8af79c48fcc488588519c0f685a9882089110709a575fd9133f26d582'}

def classify(piece,triangle,core_distance,distance,core,skin,continuation,left_names,regions):
 if not np.isfinite([core_distance,distance]).all() or distance<=0 or core_distance<=0:raise ValueError('Finite positive actual ray distances required')
 if piece==core or distance>=core_distance-1e-6:return 'target_core_first'
 if piece in left_names:return 'other_actor_first'
 if piece==continuation:return 'own_forearm_first'
 if piece==skin:return 'own_forearm_first' if int(regions[triangle])==102 else 'own_other_skin_first'
 return 'other_geometry_first'

def observe(session,layer,geometry):
 from mathutils import Vector
 from mathutils.bvhtree import BVHTree
 from data import action4_formal_composition as router
 layer.verifier(session.scene)
 observed=router.load_fixed('actual_presence_ref','formal_attribution_presence').inventory(session,layer.names)
 roles=observed['object_by_semantic_role'];core=roles['right/core'];skin=roles['right/skin'];continuation=roles['right/continuation'];left={v for k,v in roles.items() if k.startswith('left/')}
 with np.load(router.backend()._ref(SEMANTIC_REF),allow_pickle=False) as archive:
  source_faces=archive['faces'].copy();regions=archive['source_regions'].copy()
 data={};trees={};signatures={}
 for obj in session.scene.objects:
  if obj.type=='MESH' and pixels._camera_render_enabled(session.scene,obj):
   V,F=assembly.world_geometry(obj);data[obj.name]=(V,F);trees[obj.name]=BVHTree.FromPolygons(V.tolist(),F.tolist(),all_triangles=True,epsilon=0);signatures[obj.name]=env.digest({'vertices_world_m':V.tolist(),'triangles':F.tolist()})
 if skin not in data or not np.array_equal(data[skin][1],source_faces) or regions.shape!=(len(source_faces),):raise ValueError('Exact evaluated right-skin triangle order must match frozen source_regions')
 g=geometry.get('2',geometry.get(2));uv=np.asarray(g['points_native_px']);lo=uv.min(0);hi=uv.max(0);K=np.asarray(g['K_native']);W=np.asarray(session.parameters['W_from_capture']);eye=Vector(W[:3,3]);rays=[]
 for y in np.linspace(lo[1],hi[1],17):
  for x in np.linspace(lo[0],hi[0],17):
   direction=np.linalg.solve(K,np.array([x,y,1.]));direction/=np.linalg.norm(direction);direction=Vector(W[:3,:3]@direction)
   ch=trees[core].ray_cast(eye,direction)
   if ch[0] is None:continue
   hits=[]
   for name,tree in trees.items():
    h=tree.ray_cast(eye,direction)
    if h[0] is not None:hits.append((float(h[3]),name,int(h[2])))
   distance,name,triangle=min(hits);category=classify(name,triangle,float(ch[3]),distance,core,skin,continuation,left,regions)
   rays.append({'native_px':[float(x),float(y)],'core_distance_m':float(ch[3]),'nearest_distance_m':distance,'nearest_piece':name,'nearest_triangle':triangle,'source_region_id':int(regions[triangle]) if name==skin else None,'category':category})
 result={'schema':'action4.actual_forearm_attribution.v1','actual_scene_rays':True,'instance_id':2,'grid':[17,17],'core':core,'skin':skin,'continuation':continuation,'left_names':sorted(left),'semantic_source_ref':SEMANTIC_REF,'evaluated_geometry_sha256_by_name':signatures,'rays':rays,'target_core_hits':len(rays),'same_actor_forearm_first_hits':sum(r['category']=='own_forearm_first' for r in rays),'entry_sha256':env.digest(session.parameters['entry']),'parameters_sha256':env.digest(session.parameters),'composition_contract_sha256':env.digest(layer.contract),'bbox_xyxy':[*lo.tolist(),*hi.tolist()],'sampled_ray_fraction_is_mask_fraction':False}
 layer.verifier(session.scene);validate(result,session.parameters,layer.contract,geometry);return result

def validate(a,p,contract,geometry):
 from data import action4_formal_composition as router
 router.backend()._ref(SEMANTIC_REF)
 if a['schema']!='action4.actual_forearm_attribution.v1' or a['actual_scene_rays'] is not True or a['instance_id']!=2 or a['grid']!=[17,17] or a['semantic_source_ref']!=SEMANTIC_REF or a['entry_sha256']!=env.digest(p['entry']) or a['parameters_sha256']!=env.digest(p) or a['composition_contract_sha256']!=env.digest(contract) or a['sampled_ray_fraction_is_mask_fraction'] is not False:raise ValueError('Actual attribution must bind current right target and full composition')
 if a['core']!=pixels.KNOWN_TARGET_PARTS[2]['core'] or a['continuation']!='right_forearm_continuation' or a['skin']!=contract['objects'][a['continuation']]['parent']:raise ValueError('Original RIGHT core/skin/continuation names must be contract-bound')
 g=geometry.get('2',geometry.get(2));uv=np.asarray(g['points_native_px']);lo=uv.min(0);hi=uv.max(0)
 if a['bbox_xyxy']!=[*lo.tolist(),*hi.tolist()]:raise ValueError('Attribution bbox differs from actual original8')
 grid={(float(x),float(y)) for y in np.linspace(lo[1],hi[1],17) for x in np.linspace(lo[0],hi[0],17)};seen=set();count=0
 with np.load(router.backend()._ref(SEMANTIC_REF),allow_pickle=False) as archive:regions=archive['source_regions'].copy()
 for r in a['rays']:
  point=tuple(r['native_px'])
  if point not in grid or point in seen or r['nearest_piece'] not in a['evaluated_geometry_sha256_by_name']:raise ValueError('Unique declared grid/evaluated object ray witness required')
  seen.add(point);idx=r['nearest_triangle']
  if type(idx) is not int or idx<0 or (r['nearest_piece']==a['skin'] and idx>=len(regions)):raise ValueError('Exact source triangle index')
  category=classify(r['nearest_piece'],idx,r['core_distance_m'],r['nearest_distance_m'],a['core'],a['skin'],a['continuation'],set(a['left_names']),regions)
  if category!=r['category'] or r['source_region_id']!=(int(regions[idx]) if r['nearest_piece']==a['skin'] else None):raise ValueError('Nearest-hit semantic attribution differs')
  count+=category=='own_forearm_first'
 if a['target_core_hits']!=len(seen) or a['same_actor_forearm_first_hits']!=count or count<=0:raise ValueError('At least one actual own-forearm first hit required')
 return True
