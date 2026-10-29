"""Native CoTracker3 core adapter. Raw tracks are not geometry observations.

No wrapper-side hidden resize, forced query overwrite, GT, or CAD XYZ input.
"""
from dataclasses import dataclass
import hashlib,json,os,sys,subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.action4_phase2_tracking_contracts import (AnchorKey,WindowIdentity,CropTransform,support_queries,
    query_local_times,reverse_query_times,merge_bidirectional)


@dataclass(frozen=True)
class TrackWindowRequest:
    window:WindowIdentity
    frames:tuple
    crop:CropTransform
    anchor_ids:tuple
    query_frame_ids:tuple
    query_uv_native:np.ndarray
    pixel_space_id:str
    query_role:str='material_candidates'
    time_base:str|None=None


def _frozen_core_state(core):
    """Identity/version of registered parameters and buffers, without copying weights."""
    return tuple((kind,name,tensor.data_ptr(),tensor._version,str(tensor.device),
                  str(tensor.dtype),tuple(tensor.shape),tensor.requires_grad)
        for kind,items in (("parameter",core.named_parameters()),("buffer",core.named_buffers()))
        for name,tensor in items)


def _check_frozen_core(core,model_identity):
    if getattr(core,"_action4_native_identity",None)!=model_identity or not model_identity:
        raise ValueError("Native core/source binding mismatch")
    if any(module.training for module in core.modules()) or any(p.requires_grad for p in core.parameters()):
        raise ValueError("Native CoTracker must retain frozen eval mode")
    expected=getattr(core,"_action4_frozen_state",None)
    if expected is None or _frozen_core_state(core)!=expected:
        raise ValueError("Frozen native parameters or buffers changed")


def _load_core(config,device):
    cfg=json.loads(contained_path(R,config,exists=True).read_text())
    if device=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU loader requires empty CUDA visibility')
    if (cfg.get('model_hw'),cfg.get('window_len'),cfg.get('stride'),cfg.get('corr_radius'))!=([384,512],60,4,3):raise ValueError('Fixed native architecture/resize contract changed')
    repo=contained_path(R,cfg['source_root'],exists=True)
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if head!=cfg['source_commit'] or subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Official source revision changed')
    weight=contained_path(R,cfg['weight'],exists=True)
    with weight.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=cfg['weight_sha256']:raise ValueError('CoTracker weight identity mismatch')
    sys.path.insert(0,str(repo))
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    import cotracker.models.core.cotracker.cotracker3_offline as module
    if not Path(module.__file__).resolve().is_relative_to(repo):raise ValueError('Mixed CoTracker source imports')
    core=CoTrackerThreeOffline(stride=cfg['stride'],corr_radius=cfg['corr_radius'],window_len=cfg['window_len'])
    state=torch.load(weight,map_location='cpu',weights_only=True)
    if 'model' in state:state=state['model']
    core.load_state_dict(state,strict=True);core=core.to(device).eval().requires_grad_(False)
    core._action4_native_identity={'source_commit':head,'weight_sha256':actual,'architecture':'CoTrackerThreeOffline','constructor':{'window_len':60,'stride':4,'corr_radius':3},'model_hw':[384,512]}
    core._action4_native_identity['execution_device']=str(device)
    core._action4_weight_versions=tuple(p._version for p in core.parameters())
    core._action4_frozen_state=_frozen_core_state(core)
    return core,cfg


def load_cpu_core(config):
    return _load_core(config,'cpu')


def load_gpu_core(config,allocation):
    """Explicit current resource admission before CUDA initialization."""
    from model.training.action4_resource_policy import validate_device_policy
    r=allocation
    required={'physical_gpu','gpu_uuid','cpu_affinity','allocator_gib','minimum_free_mib','project_total_cpu'}
    if not isinstance(r,dict) or set(r)!=required:raise ValueError('Explicit CoTracker GPU allocation required')
    if type(r['physical_gpu']) is not int or r['physical_gpu'] not in (4,5,6,7):
        raise ValueError('Only physical GPU4-7 authorized')
    admitted=validate_device_policy({'training_gpus':[],'eval_gpus':[r['physical_gpu']]},R)
    if not isinstance(r['gpu_uuid'],str) or not r['gpu_uuid'].startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu_uuid']:
        raise ValueError('Single physical GPU UUID visibility required')
    cpus=r['cpu_affinity']
    if (not isinstance(cpus,list) or not cpus or any(type(x) is not int for x in cpus)
            or len(set(cpus))!=len(cpus) or not set(cpus)<=set(range(24,96))
            or set(os.sched_getaffinity(0))!=set(cpus)
            or type(r['project_total_cpu']) is not int or not len(cpus)<=r['project_total_cpu']<=72):
        raise ValueError('Explicit bounded CPU affinity required')
    if type(r['allocator_gib']) is not int or not 1<=r['allocator_gib']<=12 or type(r['minimum_free_mib']) is not int or r['minimum_free_mib']<r['allocator_gib']*1024+8192:
        raise ValueError('GPU allocator/headroom budget invalid')
    info=subprocess.check_output(['nvidia-smi','-i',str(r['physical_gpu']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if len(info)!=2 or info[0].strip()!=r['gpu_uuid'] or int(info[1])<r['minimum_free_mib']:
        raise ValueError('GPU mapping or current headroom mismatch')
    if torch.cuda.device_count()!=1:raise ValueError('Exactly one visible GPU required')
    torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(r['allocator_gib']*1024**3/total,0)
    core,cfg=_load_core(config,'cuda:0')
    core._action4_resource_admission={'authority':admitted['authority'],'allocation':dict(r)}
    return core,cfg


class WindowImageReader:
    """Process-local verified file identities; image decode remains explicit."""
    def __init__(self):self.files={};self.pid=os.getpid()
    @staticmethod
    def stat(p):
        s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def read(self,row):
        if self.pid!=os.getpid():self.files.clear();self.pid=os.getpid()
        p=contained_path(R,row['image'],exists=True);key=(str(p),row['image_sha256']);before=self.stat(p)
        if self.files.get(key)!=before:
            with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if self.stat(p)!=before or actual!=row['image_sha256']:raise ValueError('Frame changed or hash mismatch')
            self.files[key]=before
        with Image.open(p) as image:rgb=np.asarray(image.convert('RGB')).copy()
        if self.stat(p)!=before:raise ValueError('Frame changed while decoding')
        return rgb


def prepare_video(request,reader):
    if not isinstance(request,TrackWindowRequest) or not isinstance(request.window,WindowIdentity) or not isinstance(request.crop,CropTransform):raise ValueError('Typed window/crop required')
    rows=request.frames;ids=tuple(r['frame_id'] for r in rows)
    if not rows or ids[0]<request.window.start or ids[-1]>=request.window.end:raise ValueError('Frame interval mismatch')
    query_local_times(ids,())
    if any(type(r.get('pts')) is not int for r in rows):raise ValueError('Finite integer source PTS ticks required; bool/float/string rejected')
    if any(b['pts']<=a['pts'] for a,b in zip(rows,rows[1:])):raise ValueError('Increasing source PTS required')
    if not isinstance(request.time_base,str):raise ValueError('Explicit source time_base required')
    try:time_base=Fraction(request.time_base)
    except (ValueError,ZeroDivisionError):raise ValueError('Valid rational source time_base required')
    if time_base<=0:raise ValueError('Positive source time_base required')
    if not isinstance(request.pixel_space_id,str) or not request.pixel_space_id:raise ValueError('Pixel domain identity required')
    c=request.crop;frames=[];shape=None
    for row in rows:
        rgb=reader.read(row)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Native RGB uint8 required')
        if shape is None:shape=rgb.shape[:2]
        if shape!=rgb.shape[:2]:raise ValueError('Changing native frame geometry')
        patch=np.asarray(Image.fromarray(rgb).crop((c.x0,c.y0,c.x0+c.width,c.y0+c.height))).copy()
        tensor=torch.from_numpy(patch).permute(2,0,1)[None].float()
        frames.append(F.interpolate(tensor,size=(384,512),mode='bilinear',align_corners=True)[0])
    return torch.stack(frames)[None],ids,shape


def track_window_raw(core,request,*,reader=None,iters=6,model_identity,allow_test_backend=False):
    if type(allow_test_backend) is not bool:raise ValueError("Explicit test-backend flag required")
    if not allow_test_backend:
        _check_frozen_core(core,model_identity)
    if reader is not None and not isinstance(reader,WindowImageReader):raise ValueError("Verified native image reader required")
    if not isinstance(request,TrackWindowRequest):raise ValueError('Typed tracking window request required')
    if len(request.frames)>120 or len(request.anchor_ids)>96:raise ValueError('Registered120-frame/96-material window budget exceeded')
    if request.query_role not in ("material_candidates","engineering_input_not_material_seed"):raise ValueError("Explicit query role required")
    if type(iters) is not int or iters<1:raise ValueError('Positive native iteration count required')
    video,frame_ids,(height,width)=prepare_video(request,reader or WindowImageReader())
    n=len(request.anchor_ids)
    if not n or len(set(request.anchor_ids))!=n or any(not isinstance(k,AnchorKey) or k.track_id!=request.window.track_id for k in request.anchor_ids):raise ValueError('Unique same-track material IDs required')
    if len({(k.track_id,k.anchor_id) for k in request.anchor_ids})!=n:raise ValueError('Multiple generations for one material in window')
    if len(request.query_frame_ids)!=n:raise ValueError('One query frame per anchor required')
    times=query_local_times(frame_ids,request.query_frame_ids);native=np.asarray(request.query_uv_native,float)
    if native.shape!=(n,2) or not np.isfinite(native).all():raise ValueError('Finite native query coordinates required')
    xy=request.crop.to_model(native)
    if ((native<[0,0])|(native>[width-1,height-1])).any() or ((xy<[0,0])|(xy>[511,383])).any():raise ValueError('Material query in padding or outside crop')
    support=support_queries(request.crop,width,height);queries=np.concatenate([np.column_stack([times,xy]),support],axis=0)
    q=torch.tensor(queries,dtype=torch.float32)[None];back=q.clone();back[:,:,0]=len(frame_ids)-1-back[:,:,0]
    device=torch.device('cpu') if allow_test_backend else next(core.parameters()).device
    video=video.to(device);q=q.to(device);back=back.to(device)
    with torch.inference_mode():
        forward=core(video=video,queries=q,iters=iters,is_train=False)
        backward=core(video=video.flip(1),queries=back,iters=iters,is_train=False)
    def unpack(output):
        if not isinstance(output,(tuple,list)) or len(output)!=4 or output[3] is not None:raise ValueError('Official inference must return coords/visibility/confidence/None')
        arrays=[];total=len(queries)
        for value,shape in zip(output[:3],[(1,len(frame_ids),total,2),(1,len(frame_ids),total),(1,len(frame_ids),total)]):
            if not torch.is_tensor(value) or tuple(value.shape)!=shape or not value.is_floating_point():raise ValueError('Native output shape/dtype mismatch')
            arrays.append(value.detach().cpu().numpy()[0].copy())
        return tuple(arrays)
    if not allow_test_backend:_check_frozen_core(core,model_identity)
    f,b=unpack(forward),unpack(backward)
    # Preserve nonfinite values for diagnostics; never turn them into zero/absent.
    scale=np.array([(request.crop.width-1)/511,(request.crop.height-1)/383]);offset=np.array([request.crop.x0,request.crop.y0])
    fn=(f[0][:,:n]*scale+offset,f[1][:,:n],f[2][:,:n]);bn=(b[0][:,:n]*scale+offset,b[1][:,:n],b[2][:,:n])
    merged=merge_bidirectional(fn,bn,times,native)
    numerical=bool(np.isfinite(f[0]).all() and np.isfinite(b[0]).all() and all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in (*f[1:],*b[1:])))
    identity={'adapter_protocol':'action4.cotracker.core_bridge.v1','query_role':request.query_role,'test_backend':allow_test_backend,'frame_PTS':[r['pts'] for r in request.frames],'time_base':str(Fraction(request.time_base)),
        'window':request.window.__dict__,'frame_ids':list(frame_ids),'frame_SHA':[r['image_sha256'] for r in request.frames],
        'pixel_space_id':request.pixel_space_id,'crop':request.crop.__dict__,'anchor_ids':[k.__dict__ for k in request.anchor_ids],
        'query_frame_ids':list(request.query_frame_ids),'query_uv_native':native.tolist(),'model_identity':model_identity,'iters':iters}
    snapshot=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'status':'RAW_INFERENCE_COMPLETE' if numerical else 'RAW_INFERENCE_HAS_INVALID_VALUES','snapshot_identity':snapshot,'identity':identity,
        'raw_forward_model':f,'raw_backward_model':b,'queries_forward_model':queries,'queries_backward_model':back.detach().cpu().numpy()[0],
        'merged':merged,'support_query_count':len(support),'support_has_CAD_XYZ':False,'material_query_count':n,
        'query_role':request.query_role,'geometry_observations_accepted':False,'native_model_range':[float(video.min()),float(video.max())],
        'frame_PTS':[r['pts'] for r in request.frames],'time_base':str(Fraction(request.time_base)),'official_core_called_directly':not allow_test_backend,'test_backend':allow_test_backend,'wrapper_query_override_used':False}
