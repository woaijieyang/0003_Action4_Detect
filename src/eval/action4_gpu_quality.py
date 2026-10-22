"""Review-only GPU3 quality aid. No CUDA work at import. No acceptance authority.
Original native/canonical/mask/support gates remain in explicit validate_record.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import os
import threading

GPU3_UUID = 'GPU-510dc520-5a9d-1317-019d-de5ed02c1b37'
ORIGINAL_AID_SHA = 'c847e052669fc90ac42cc37ba0391d55201b34e8fa246341fce3c2cdf82d5427'
ROOT = Path('/data/zy/vln/a-test')


def load_original():
    from eval import action4_corner_checks as original
    p = Path(original.__file__).resolve()
    if p != ROOT/'src/eval/action4_corner_checks.py' or hashlib.sha256(p.read_bytes()).hexdigest() != ORIGINAL_AID_SHA:
        raise ValueError('Exact original reviewed corner aid required')
    return original


class FP64Math:
    """Torch numerical kernel; CPU device is for test oracle only, never auto fallback."""
    def __init__(self, device, chunk_pixels=250000):
        import torch
        if type(chunk_pixels) is not int or not 1 <= chunk_pixels <= 1000000:
            raise ValueError('Bounded positive chunk_pixels required')
        self.torch = torch
        self.device = torch.device(device)
        self.chunk_pixels = chunk_pixels
        self.projection_calls = 0
        self.hull_calls = 0
        self.hull_pixels = 0
        self.hull_chunks = 0

    def tensor(self, value):
        return self.torch.as_tensor(value, dtype=self.torch.float64, device=self.device)

    def project8(self, g):
        import numpy as np
        torch = self.torch
        xyz = np.asarray(g['points_object_mm'], dtype=float)
        T = np.asarray(g['object_to_world_m'], dtype=float)
        C = np.asarray(g['camera_to_world_blender_m'], dtype=float)
        K = np.asarray(g['K_native'], dtype=float)
        if xyz.shape != (8,3) or T.shape != (4,4) or C.shape != (4,4) or K.shape != (3,3) or not all(np.isfinite(x).all() for x in (xyz,T,C,K)):
            raise ValueError('Finite canonical matrices required')
        points = torch.cat((self.tensor(xyz)/1000., torch.ones((8,1),dtype=torch.float64,device=self.device)),dim=1)
        camera = (torch.linalg.inv(self.tensor(C)) @ self.tensor(T) @ points.T)[:3].T * self.tensor([1.,-1.,-1.])
        near = float(g['near_plane_m'])
        if not np.isfinite(near) or near <= 0 or not bool(torch.isfinite(camera).all().item()) or not bool((camera[:,2] > near).all().item()):
            raise ValueError('Original8 not near-plane projectable')
        homogeneous = (self.tensor(K) @ camera.T).T
        uv = homogeneous[:,:2]/homogeneous[:,2,None]
        self.projection_calls += 1
        return uv.detach().cpu().numpy(), camera[:,2].detach().cpu().numpy()

    def outside_hull(self, xy, equations):
        import numpy as np
        torch = self.torch
        xy = np.asarray(xy,dtype=float)
        equations = np.asarray(equations,dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or equations.ndim != 2 or equations.shape[1] != 3 or not np.isfinite(xy).all() or not np.isfinite(equations).all():
            raise ValueError('Finite pixel centres and hull halfspaces required')
        eq = self.tensor(equations)
        bad = torch.zeros((),dtype=torch.int64,device=self.device)
        for begin in range(0,len(xy),self.chunk_pixels):
            points = self.tensor(xy[begin:begin+self.chunk_pixels])
            bad += ((points @ eq[:,:2].T + eq[:,2]) > 1e-6).any(dim=1).sum()
            self.hull_chunks += 1
        self.hull_calls += 1
        self.hull_pixels += len(xy)
        return int(bad.item())


class GPUAid:
    """One worker owns one CUDA context; exact UUID visibility, no CPU fallback."""
    def __init__(self, *, implementation_ref, chunk_pixels=250000):
        source = Path(__file__).resolve()
        actual_ref = {'path':str(source.relative_to(ROOT)), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        if implementation_ref != actual_ref:
            raise ValueError('Explicit reviewed GPU quality implementation identity required')
        import torch
        if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU3_UUID:
            raise ValueError('Quality worker must expose only authorized GPU3 UUID')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one CUDA device required; no fallback')
        properties = torch.cuda.get_device_properties(0)
        actual = getattr(properties, 'uuid', None)
        if actual is None:
            raise RuntimeError('Runtime lacks CUDA property UUID; add a reviewed UUID query before execution')
        normalized = str(actual).lower().removeprefix('gpu-')
        if normalized != GPU3_UUID.lower().removeprefix('gpu-'):
            raise ValueError('Visible physical GPU UUID differs from authorized GPU3')
        if torch.get_num_threads() != 2:
            raise ValueError('Each quality worker must have exactly two torch CPU threads')
        self.original = load_original()
        self.math = FP64Math('cuda:0', chunk_pixels)
        self.owner = (os.getpid(), threading.get_ident())
        self.device_name = properties.name
        self.DEPENDENCIES = deepcopy(self.original.DEPENDENCIES)
        self.DEPENDENCIES[implementation_ref['path']] = implementation_ref['sha256']
        self.DEPENDENCIES['src/eval/action4_corner_checks.py'] = ORIGINAL_AID_SHA
        self.ReadSet = self.original.ReadSet
        self.passed_records = 0
        self.zero_target_records = 0

    def guard(self):
        if self.owner != (os.getpid(),threading.get_ident()):
            raise ValueError('GPU quality context cannot cross process or thread')

    def check_presence(self, record, entry):
        self.guard()
        return self.original.check_presence(record,entry)

    def rel(self, value): return self.original.rel(value)
    def command_value(self, command, key, default=None): return self.original.command_value(command,key,default)

    def project8(self, geometry):
        self.guard()
        return self.math.project8(geometry)

    def validate_record(self, reader, record, data):
        self.guard()
        value = validate_record_gpu(reader,record,data,engine=self.math,original=self.original)
        self.passed_records += 1
        self.zero_target_records += int(not record['instances'])
        return value

    def receipt(self):
        self.guard()
        self.math.torch.cuda.synchronize(self.math.device)
        return {'schema':'action4.gpu_quality_compute.v1','gpu_uuid':GPU3_UUID,
                'device_name':self.device_name,'dtype':'torch.float64',
                'projection_calls':self.math.projection_calls,
                'hull_calls':self.math.hull_calls,'hull_pixels':self.math.hull_pixels,
                'hull_chunks':self.math.hull_chunks,'chunk_pixels':self.math.chunk_pixels,
                'passed_records':self.passed_records,'zero_target_records':self.zero_target_records,
                'zero_target_has_no_projection_or_hull_work':True,
                'training_accepted':False,'visual_review_still_required':True}


def check_or_request(aid, reader, record, data, *, entry, manifest_ref):
    """Return a review request on a domain rejection; never mutate/delete/rerender.
    Operational CUDA/IO errors propagate and are not counted as bad samples.
    """
    aid.guard()  # Ownership failures are operational, never a sample rejection.
    try:
        aid.check_presence(record,entry)
        result = aid.validate_record(reader,record,data)
        return {'status':'PASS_EXISTING_NUMERICAL_QUALITY_GATES_VISUAL_PENDING',
                'result':result,'training_accepted':False}
    except ValueError as exc:
        return {'status':'REJECT_REVIEW_AND_REPLACEMENT_REQUEST',
                'manifest_ref':deepcopy(manifest_ref),'expected_entry':deepcopy(entry),
                'observed_sample_id':record.get('sample_id'),'reason':str(exc),
                'required_same_split':entry['split'],'required_same_group_id':entry['group_id'],
                'required_same_presence':deepcopy(entry['target_presence']),
                'original_evidence_must_be_retained':True,
                'replacement_requires_root_authorized_attempt_or_plan_revision':True,
                'automatic_acceptance':False,'training_accepted':False}


def validate_record_gpu(reader,record,data,*,engine,original):
 bounded=original.bounded
 draw8=original.draw8
 project8=engine.project8
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
   bad=engine.outside_hull(xy,equations)
   outside[key]=bad
   if bad:raise ValueError('Target mask pixel centres outside original8 hull: '+str((ident,key,bad)))
  draw8(overlay,uv,(0,255,80) if ident==1 else (255,175,0))
  rows.append({'instance_id':ident,'independent_points_native_px':uv.tolist(),'depth_m':depth.tolist(),'point_inframe':inframe,'points_clamped':False,'original8_base_layer_render_equal':True,'independent_projection_max_error_px':float(np.abs(uv-native).max()),'legacy_projection_max_error_px':float(np.abs(uv-np.asarray(row['corners_px'])).max()),'outside_hull_pixels':outside,'visible_fraction':measured['area_visible_px']/measured['area_inframe_amodal_px'],'visible_area_fullinput_px':measured['visible_area_at_fullinput_px']})
 return overlay,rows,semantic
