"""Reusable original8/mask checks promoted without changing checked algorithms.
Origin: reviewed native aid SHA785a1d93fde8fd5b3f744578c62499b36364fc3ee38d711bdd197f7075444b70.
The historical file remains evidence; this module has no executable dependency on it.
"""
from pathlib import Path
import json,hashlib
ROOT=Path('/data/zy/vln/a-test')

DEPENDENCIES={'src/data/action4_mask_semantics.py': '7dad609269ee452fb38965751b83a9623db78688d1fe5aeb8b6d63b5d4633181', 'src/data/action4_adapter.py': 'df78347afbf0a3bb2ab004b4b80fbd76e99760c03d0f39450c24f5902f89fa7c', 'src/data/action4_global_render.py': '40584a0665b5d9173ce64446d39bd79bf2bb8e0b187ed1e8bdbcae4036223392'}


EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


ENTRY_KEYS=('sample_id','index','seed','scene_family','target_count','target_presence','group_id','background_source_id','background_seed','split')


def bounded(value):
 p=Path(value)
 if '..' in p.parts:raise ValueError('Traversal forbidden')
 p=(p if p.is_absolute() else ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
 return p


def rel(value):return str(bounded(value).relative_to(ROOT))


def sha(raw):return hashlib.sha256(raw).hexdigest()


class ReadSet:
 def __init__(self):self.refs={}
 def data(self,value,expected=None):
  p=bounded(value);raw=p.read_bytes();h=sha(raw);name=rel(p)
  if expected is not None and h!=expected:raise ValueError('SHA mismatch '+name)
  if name in self.refs and self.refs[name]!=h:raise ValueError('Input drift '+name)
  self.refs[name]=h;return raw
 def json(self,value,expected=None):return json.loads(self.data(value,expected))
 def close(self):
  for name,h in self.refs.items():
   if sha(bounded(name).read_bytes())!=h:raise ValueError('Input changed '+name)
  return dict(sorted(self.refs.items()))


def expected_ids(entry):
 presence=entry.get('target_presence');count=entry.get('target_count')
 if not isinstance(presence,list) or len(presence)!=2 or any(type(v) is not bool for v in presence) or type(count) is not int or count!=sum(presence):raise ValueError('Exact presence/count')
 if entry.get('split')=='test' and count!=2:raise ValueError('Formal test must remain dual')
 return [i for i,on in enumerate(presence,1) if on]


def check_presence(record,entry):
 if any(record.get(k)!=entry.get(k) for k in ENTRY_KEYS):raise ValueError('Record differs from frozen entry')
 ids=expected_ids(entry)
 if record.get('planned_target_count')!=len(ids) or record.get('excluded_instances')!=[] or [x.get('instance_id') for x in record.get('instances',[])]!=ids:raise ValueError('Original IDs/presence/excluded mismatch')
 if not ids and any(k in record for k in ('mask_visible','mask_amodal_expanded','canonical8_geometry')):raise ValueError('Zero target may not carry dummy target evidence')
 return ids


def clip_segment(a,b,width,height):
 """Liang-Barsky display clipping; leaves original point arrays unchanged."""
 x,y=map(float,a);xx,yy=map(float,b);dx,dy=xx-x,yy-y;t0,t1=0.,1.
 for p,q in ((-dx,x),(dx,width-1-x),(-dy,y),(dy,height-1-y)):
  if p==0:
   if q<0:return None
   continue
  t=q/p
  if p<0:t0=max(t0,t)
  else:t1=min(t1,t)
  if t0>t1:return None
 return ((x+t0*dx,y+t0*dy),(x+t1*dx,y+t1*dy))


def project8(g):
 import numpy as np
 xyz=np.asarray(g['points_object_mm'],dtype=float)
 T=np.asarray(g['object_to_world_m'],dtype=float);C=np.asarray(g['camera_to_world_blender_m'],dtype=float);K=np.asarray(g['K_native'],dtype=float)
 if xyz.shape!=(8,3) or T.shape!=(4,4) or C.shape!=(4,4) or K.shape!=(3,3) or not all(np.isfinite(x).all() for x in (xyz,T,C,K)):raise ValueError('Finite canonical matrices required')
 camera=(np.linalg.inv(C)@T@np.c_[xyz/1000.,np.ones(8)].T)[:3].T*np.array([1.,-1.,-1.])
 near=float(g['near_plane_m'])
 if not np.isfinite(near) or near<=0 or not np.isfinite(camera).all() or not (camera[:,2]>near).all():raise ValueError('Original8 not near-plane projectable')
 homogeneous=(K@camera.T).T
 return homogeneous[:,:2]/homogeneous[:,2,None],camera[:,2]


def draw8(image,uv,color):
 from PIL import ImageDraw
 draw=ImageDraw.Draw(image)
 for a,b in EDGES:
  line=clip_segment(uv[a],uv[b],*image.size)
  if line is not None:draw.line(line,fill=color,width=5)
 for i,(x,y) in enumerate(uv):
  if 0<=x<image.width and 0<=y<image.height:
   draw.ellipse((x-4,y-4,x+4,y+4),fill=color);draw.text((x+6,y+2),str(i),fill=color)


def command_value(command,key,default=None):
 if command.count(key)>1:raise ValueError('Ambiguous launch '+key)
 if key not in command:return default
 return command[command.index(key)+1]


def validate_record(reader,record,data):
 import numpy as np
 from PIL import Image
 from scipy.spatial import ConvexHull
 from data.action4_adapter import DataAdapter
 from data import action4_mask_semantics as masks
 DataAdapter(ROOT).validate_sample(record,check_image=False)
 if (record['width'],record['height'])!=(3248,2464):raise ValueError('Native3248x2464 required')
 imagepath=bounded(record['image'])
 if not imagepath.is_relative_to(data):raise ValueError('RGB outside its own shard')
 reader.data(imagepath,record['rgb_sha256'])
 with Image.open(imagepath) as im:
  if im.format!='PNG' or im.size!=(record['width'],record['height']):raise ValueError('Actual native PNG required')
  im.load();overlay=im.convert('RGB')
 manifest=record['canonical_source_manifest'];source=reader.json(manifest['path'],manifest['sha256'])
 if source['corners_3d_mm']!=record['corners_3d_mm']:raise ValueError('Original8 differs from immutable asset manifest')
 # Track exact mask bytes before shared helper independently decodes/remeasures.
 for row in record['instances']:
  for key in ('mask_visible','mask_amodal_expanded'):
   if not bounded(row[key]).is_relative_to(data):raise ValueError('Target mask outside its own shard')
   reader.data(row[key],row[key+'_sha256'])
 semantic=masks.verify_record_masks(record)
 original=record['pose_controls_proof']['geometry']
 layered=record['pose_controls_proof']['composition']['preflight_geometry']
 ids={str(x['instance_id']) for x in record['instances']}
 if set(map(str,original))!=ids or set(map(str,layered))!=ids:raise ValueError('Exact original/layer probe presence required')
 if not record['instances']:
  if semantic['measurements_by_instance_id'] or list((data/'masks').glob(record['sample_id']+'_*')):raise ValueError('Zero target has dummy/undeclared masks')
 rows=[]
 for row in record['instances']:
  ident=row['instance_id'];g=row['canonical8_geometry']
  if g['instance_id']!=ident or g['object_id']!=1 or g['corner_ids']!=list(range(8)) or g['points_object_mm']!=source['corners_3d_mm'] or g['canonical_source_manifest']!=manifest or g['corner_order']!=source['corner_order']:raise ValueError('Immutable ordered canonical8 mismatch')
  if g['image_size_wh']!=[record['width'],record['height']] or g['K_native']!=record['camera']['K'] or g['camera_to_world_blender_m']!=record['camera']['camera_to_world_m'] or g['object_to_world_m']!=row['object_to_world_m']:raise ValueError('Record/canonical camera or object matrix mismatch')
  uv,depth=project8(g);native=np.asarray(g['points_native_px'])
  if not np.allclose(uv,native,rtol=0,atol=1e-6) or not np.allclose(uv,row['corners_px'],rtol=0,atol=.003):raise ValueError('Independent original8 reprojection failed')
  for probe in (original[str(ident)],layered[str(ident)]):
   probe_uv,_=project8(probe)
   if not np.allclose(probe_uv,uv,rtol=0,atol=1e-6) or not np.allclose(probe['points_native_px'],uv,rtol=0,atol=1e-6):raise ValueError('Original8 moved between base/layer/render')
  if row['corners_px']!=g['legacy_corners_px'] or row['corner_visible']!=g['legacy_corner_visible']:raise ValueError('Legacy source labels changed')
  inframe=((uv[:,0]>=-.5)&(uv[:,0]<=record['width']-.5)&(uv[:,1]>=-.5)&(uv[:,1]<=record['height']-.5)).tolist()
  if g['point_inframe']!=inframe:raise ValueError('Stored inframe flags differ from original coordinates')
  measured=semantic['measurements_by_instance_id'][str(ident)]
  if measured['ignore'] or measured['area_inframe_amodal_px']<=0 or measured['area_visible_px']/measured['area_inframe_amodal_px']<.1 or measured['visible_area_at_fullinput_px']<16:raise ValueError('Unchanged10percent/16px support failed')
  equations=ConvexHull(uv).equations;outside={}
  for key in ('mask_visible','mask_amodal_expanded'):
   array=masks._decode_mask(bounded(row[key]));y,x=np.nonzero(array);xy=np.c_[x,y].astype(float)
   if key=='mask_amodal_expanded':xy+=np.asarray(row['amodal_expanded_origin_px'])
   bad=0
   for i in range(0,len(xy),50000):bad+=int(((xy[i:i+50000]@equations[:,:2].T+equations[:,2])>1e-6).any(axis=1).sum())
   outside[key]=bad
   if bad:raise ValueError('Target mask pixel centres outside original8 hull: '+str((ident,key,bad)))
  draw8(overlay,uv,(0,255,80) if ident==1 else (255,175,0))
  rows.append({'instance_id':ident,'independent_points_native_px':uv.tolist(),'depth_m':depth.tolist(),'point_inframe':inframe,'points_clamped':False,'original8_base_layer_render_equal':True,'independent_projection_max_error_px':float(np.abs(uv-native).max()),'legacy_projection_max_error_px':float(np.abs(uv-np.asarray(row['corners_px'])).max()),'outside_hull_pixels':outside,'visible_fraction':measured['area_visible_px']/measured['area_inframe_amodal_px'],'visible_area_fullinput_px':measured['visible_area_at_fullinput_px']})
 return overlay,rows,semantic

