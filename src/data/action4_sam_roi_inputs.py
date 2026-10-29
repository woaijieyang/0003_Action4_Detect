"""Prediction-only native SAM mask -> existing RGB ROI coordinate contract.

No SAM calls, no GT fallback, no model imports, no training integration.
Caller injects the existing MaskAux constructor and FeatureIdentity objects.
"""
from dataclasses import dataclass,field
from pathlib import Path
import hashlib,json,os
import numpy as np
from .action4_adapter import DataAdapter
from .action4_dataset import prepare_rois
from .action4_paths import PROJECT_ROOT,contained_path

SCHEMA='action4.predicted_sam.native.v2'
_VERIFIED_TOKEN=object()
SOURCE_KEYS=('b0_model_sha256','b0_source_sha256','yolo_model_sha256','yolo_source_sha256',
             'yolo_prompt_sha256','sam_model_sha256','sam_source_sha256','sam_prompt_sha256','postprocess_sha256')

def _digest(value):
    if not isinstance(value,str) or len(value)!=64 or value=='0'*64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected explicit lowercase SHA256 identity')

def array_digest(array):
    a=np.ascontiguousarray(array)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes());return h.hexdigest()

def identity_key(identity):
    values=tuple(getattr(identity,k,None) for k in ('image_id','instance_id','roi_id','pixel_space_id'))
    if any(not isinstance(x,str) or not x.strip() for x in values):raise ValueError('Explicit image/instance/ROI/pixel-space identity required')
    return values

def _sources(sources):
    if not isinstance(sources,dict):raise ValueError('Unknown or incomplete source dependency schema')
    keys=set(SOURCE_KEYS)
    if sources.get('upstream_mode') in ('b0_only_native_receipt_v1','b0_completed_native_receipt_v2','b0_completed_native_rgb_variant_receipt_v1'):
        keys=keys-{'yolo_model_sha256','yolo_source_sha256','yolo_prompt_sha256'}|{'upstream_mode'}
    if set(sources)!=keys:raise ValueError('Unknown or incomplete source dependency schema')
    for key,value in sources.items():
        if key!='upstream_mode':_digest(value)

@dataclass(frozen=True)
class ImageFrame:
    image_id:str
    pixels:np.ndarray
    pixel_space_id:str
    source_sha256:str
    variant_id:str
    variant_sha256:str  # array_digest of the ACTUAL pixels, not a copied file hash

@dataclass(frozen=True)
class NativeMaskIdentity:
    image_id:str
    instance_id:str
    pixel_space_id:str  # native/source domain; NEVER draw-time roi_id


def native_key(identity):
    if not isinstance(identity,NativeMaskIdentity):raise ValueError('Native cache requires NativeMaskIdentity, not ROI identity')
    values=(identity.image_id,identity.instance_id,identity.pixel_space_id)
    if any(not isinstance(x,str) or not x for x in values):raise ValueError('Native identity fields required')
    return values


def _immutable_array(array):
    a=np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _bytes_backed(array):
    if not isinstance(array,np.ndarray) or array.flags.writeable:return False
    owner=array
    while isinstance(owner,np.ndarray) and owner.base is not None:owner=owner.base
    return type(owner) is bytes


@dataclass(frozen=True)
class VerifiedImageFrame(ImageFrame):
    verified_shape:tuple
    validation_receipt:str
    _token:object=field(default=None,init=False,repr=False,compare=False)
    _verified_pid:object=field(default=None,init=False,repr=False,compare=False)
    def __reduce_ex__(self,protocol):
        raise TypeError('Verified native cache is process-local; verify raw inputs inside each worker')


@dataclass(frozen=True)
class VerifiedNativeSamPrediction:
    metadata_json:str
    mask:object
    verified_shape:object
    has_foreground:bool
    validation_receipt:str
    _token:object=field(default=None,init=False,repr=False,compare=False)
    _verified_pid:object=field(default=None,init=False,repr=False,compare=False)
    def __reduce_ex__(self,protocol):
        raise TypeError('Verified native cache is process-local; verify raw inputs inside each worker')
    @property
    def metadata(self):return json.loads(self.metadata_json)


def verify_image_frame(frame):
    """Validate once and copy into immutable bytes-backed pixel storage."""
    _frame(frame)
    if isinstance(frame,VerifiedImageFrame):return frame
    receipt=hashlib.sha256((frame.source_sha256+frame.variant_sha256+frame.pixel_space_id).encode()).hexdigest()
    verified=VerifiedImageFrame(frame.image_id,_immutable_array(frame.pixels),frame.pixel_space_id,
        frame.source_sha256,frame.variant_id,frame.variant_sha256,tuple(frame.pixels.shape),receipt)
    object.__setattr__(verified,'_token',_VERIFIED_TOKEN)
    object.__setattr__(verified,'_verified_pid',os.getpid())
    return verified


def verify_native_prediction(prediction,frame,native_identity,expected_sources,*,allow_synthetic=False):
    """One-time full-byte verification; reusable across different draw-time ROIs."""
    _frame(frame);_sources(expected_sources)
    _prediction(prediction,frame,native_identity,expected_sources,allow_synthetic)
    if prediction is None:raise ValueError('Missing cache is not a verified prediction')
    if isinstance(prediction,VerifiedNativeSamPrediction):return prediction
    text=json.dumps(prediction.metadata,sort_keys=True,separators=(',',':'),allow_nan=False)
    mask=None if prediction.mask is None else _immutable_array(prediction.mask)
    verified=VerifiedNativeSamPrediction(text,mask,None if mask is None else tuple(mask.shape),bool(mask is not None and mask.any()),hashlib.sha256(text.encode()).hexdigest())
    object.__setattr__(verified,'_token',_VERIFIED_TOKEN)
    object.__setattr__(verified,'_verified_pid',os.getpid())
    return verified


@dataclass(frozen=True)
class RoiRequest:
    identity:object  # caller's FeatureIdentity; data does not import model types
    image_index:int
    bounds_native:tuple
    K_native:object=None
    source_pixel_space_id:str=None

@dataclass(frozen=True)
class NativeSamPrediction:
    metadata:dict
    mask:object  # bool native HxW or None for explicit unavailable output

@dataclass(frozen=True)
class SamRoiInputs:
    rgb:object
    mask_aux:object
    A448:np.ndarray
    A112:np.ndarray
    A224:np.ndarray
    valid_image_support:object
    reasons:tuple
    source_dependencies:tuple
    K112:tuple
    K224:tuple

@dataclass(frozen=True)
class QualityTargets:
    gt_visible448:object
    supervised:object
    gt_identities:tuple

def cache_key(metadata):
    payload={k:v for k,v in metadata.items() if k!='cache_key'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def prediction_metadata(frame,identity,mask,sources,*,status='available',assigned_instance_id=None,
                        candidate_overlap=0.,reason=None,origin='predicted_sam',sam_binding=None):
    """Metadata constructor only; does not claim supplied arrays came from SAM."""
    key=native_key(identity)
    result=dict(schema=SCHEMA,native_identity=list(key),source_sha256=frame.source_sha256,
        variant_id=frame.variant_id,variant_sha256=frame.variant_sha256,native_shape=list(frame.pixels.shape[:2]),
        sources=dict(sources),mask_sha256=None if mask is None else array_digest(mask),status=status,
        assigned_instance_id=assigned_instance_id,candidate_overlap=float(candidate_overlap),reason=reason,
        origin=origin,sam_binding=sam_binding)
    result['cache_key']=cache_key(result);return result

def read_prediction_cache(path):
    """Read only owned native NPZ. Builder still verifies every identity and byte hash."""
    p=contained_path(PROJECT_ROOT,path,exists=True)
    with np.load(p,allow_pickle=False) as f:
        if set(f.files) not in ({'metadata'},{'metadata','mask'}):raise ValueError('Unexpected cache fields')
        meta=f['metadata']
        if meta.ndim!=0 or meta.dtype.kind!='U':raise ValueError('Cache metadata must be scalar UnicodeJSON')
        metadata=json.loads(meta.item());mask=f['mask'].copy() if 'mask' in f.files else None
    return NativeSamPrediction(metadata,mask)

def _frame(frame):
    if isinstance(frame,VerifiedImageFrame):
        if frame._token is not _VERIFIED_TOKEN or frame._verified_pid!=os.getpid() or not _bytes_backed(frame.pixels) or frame.pixels.shape!=frame.verified_shape or frame.pixels.dtype!=np.uint8:raise ValueError('Untrusted or mutable verified frame')
        expected=hashlib.sha256((frame.source_sha256+frame.variant_sha256+frame.pixel_space_id).encode()).hexdigest()
        if frame.validation_receipt!=expected:raise ValueError('Verified frame receipt mismatch')
        return
    if not isinstance(frame,ImageFrame) or not all(isinstance(v,str) and v for v in (frame.image_id,frame.pixel_space_id,frame.variant_id)):
        raise ValueError('Invalid frame identity')
    a=frame.pixels
    if not isinstance(a,np.ndarray) or a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<1:raise ValueError('Frame must be native uint8 RGBHWC')
    _digest(frame.source_sha256);_digest(frame.variant_sha256)
    if array_digest(a)!=frame.variant_sha256:raise ValueError('ActualRGB variant hash mismatch')

def _prediction(pred,frame,identity,sources,allow_synthetic):
    if pred is None:return None,0.,'missing_prediction_cache',None
    verified=isinstance(pred,VerifiedNativeSamPrediction)
    if verified and (pred._token is not _VERIFIED_TOKEN or pred._verified_pid!=os.getpid() or (pred.mask is not None and (not _bytes_backed(pred.mask) or pred.mask.shape!=pred.verified_shape or pred.mask.dtype!=np.bool_))):raise ValueError('Untrusted or mutable verified mask')
    if not isinstance(pred,(NativeSamPrediction,VerifiedNativeSamPrediction)):raise ValueError('Expected native prediction cache')
    if verified and pred.validation_receipt!=hashlib.sha256(pred.metadata_json.encode()).hexdigest():raise ValueError('Verified mask receipt mismatch')
    m=pred.metadata
    required={'schema','native_identity','source_sha256','variant_id','variant_sha256','native_shape','sources','mask_sha256','status','assigned_instance_id','candidate_overlap','reason','origin','sam_binding','cache_key'}
    if set(m)!=required or m['schema']!=SCHEMA:raise ValueError('Unknown cache identity schema')
    if m['cache_key']!=cache_key(m):raise ValueError('Cache metadata integrity mismatch')
    if (native_key(identity)[0],native_key(identity)[2])!=(frame.image_id,frame.pixel_space_id):raise ValueError('Cache native identity does not match source frame')
    if m['native_identity']!=list(native_key(identity)):raise ValueError('Cache native image/instance/pixel-space mismatch')
    if (m['source_sha256'],m['variant_id'],m['variant_sha256'],m['native_shape'])!=(frame.source_sha256,frame.variant_id,frame.variant_sha256,list(frame.pixels.shape[:2])):
        raise ValueError('Cache image/variant/native-shape mismatch')
    _sources(m['sources'])
    if m['sources']!=sources:raise ValueError('Frozen B0/YOLO/SAM/prompt source mismatch')
    if m['origin']!='predicted_sam' and not(allow_synthetic and m['origin']=='synthetic_fixture'):raise ValueError('GT or unregistered synthetic evidence cannot be used as predictedSAM')
    overlap=m['candidate_overlap']
    if not isinstance(overlap,(int,float)) or not np.isfinite(overlap) or not 0<=overlap<=1:raise ValueError('Invalid predicted candidate overlap')
    if pred.mask is not None:
        a=np.asarray(pred.mask)
        if a.shape!=frame.pixels.shape[:2] or a.dtype!=np.bool_:raise ValueError('Native mask must be bool and match full RGBshape')
        if not verified and array_digest(a)!=m['mask_sha256']:raise ValueError('Cached mask bytes changed')
    elif m['mask_sha256'] is not None:raise ValueError('Declared mask is missing')
    if m['status'] not in ('available','missing','ambiguous','wrong_assignment','rejected','target_visibility_unknown','inference_error','missing_response','session_lost'):raise ValueError('Unknown predicted mask status')
    if m['status']!='available':
        if not isinstance(m['reason'],str) or not m['reason']:raise ValueError('Unavailable prediction requires reason')
        return None,float(overlap),m['status']+':'+m['reason'],m
    if m['assigned_instance_id']!=native_key(identity)[1]:return None,float(overlap),'wrong_assigned_instance',m
    binding=m['sam_binding']
    if not isinstance(binding,dict) or set(binding)!={'session_id','obj_id','generation','chosen_mask_id','match_basis'} or any(v is None or str(v)=='' for v in binding.values()):
        raise ValueError('Available prediction requires explicit SAMbinding')
    if pred.mask is None:return None,float(overlap),'missing_native_mask',m
    if not (pred.has_foreground if verified else np.any(pred.mask)):return None,float(overlap),'empty_native_mask',m
    return pred.mask,float(overlap),'available',m

def _native_nearest_grid(shape_hw,A448):
    """One float32 grid, retaining the original inverse and pixel-center arithmetic."""
    import torch
    if (len(shape_hw)!=2 or any(not isinstance(n,(int,np.integer)) or isinstance(n,(bool,np.bool_)) or n<1 for n in shape_hw)):
        raise ValueError('Positive native image dimensions required')
    matrix=np.asarray(A448,float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2],[0,0,1]) or abs(np.linalg.det(matrix))<1e-12:raise ValueError('Invalid actualRGB affine')
    yy,xx=torch.meshgrid(torch.arange(448,dtype=torch.float32),torch.arange(448,dtype=torch.float32),indexing='ij')
    destination=torch.stack((xx,yy,torch.ones_like(xx)),dim=-1)
    xy=destination@torch.as_tensor(np.linalg.inv(matrix),dtype=torch.float32).T
    return torch.stack((2*(xy[...,0]+.5)/shape_hw[1]-1,2*(xy[...,1]+.5)/shape_hw[0]-1),dim=-1)


def _sample_native_nearest_on_grid(native_mask,grid):
    """Keep external finite/binary validation even when the grid is reused."""
    import torch
    from torch.nn import functional as F
    a=np.asarray(native_mask)
    if a.ndim!=2 or min(a.shape)<1 or (a.dtype!=np.bool_ and (not np.isfinite(a).all() or not np.isin(a,[0,1]).all())):raise ValueError('Finite binary native mask required')
    return F.grid_sample(torch.as_tensor(a.astype(np.float32))[None,None],grid[None],mode='nearest',padding_mode='zeros',align_corners=False)[0]


def _sample_native_valid_on_grid(shape_hw,grid):
    """Sample an internal constant through the same kernel; never materialize HxW ones.

    grid_sample only reads its input. The expanded zero-stride view is never
    mutated or exposed as a writable cache; nearest half-pixel ties stay native.
    """
    import torch
    from torch.nn import functional as F
    source=torch.ones(1,1,1,1,dtype=torch.float32).expand(1,1,*shape_hw)
    return F.grid_sample(source,grid[None],mode='nearest',padding_mode='zeros',align_corners=False)[0]


def sample_native_nearest(native_mask,A448):
    """Reuse actual RGB A448; identical inverse/pixel-center/grid_sample convention."""
    a=np.asarray(native_mask)
    if a.ndim!=2 or min(a.shape)<1:raise ValueError('Finite binary native mask required')
    return _sample_native_nearest_on_grid(a,_native_nearest_grid(a.shape,A448))


def prepare_sam_roi_inputs(frames,requests,predictions,*,expected_sources,mask_aux_factory,dropout=None,allow_synthetic=False,augmentation_policy=None,augmentation_seeds=None):
    """CPU preparation. Native predictions keyed by image/instance/source-pixel-space, never ROI or array order.

    Pass mask_aux_factory=the existing model MaskAux class from the orchestration
    layer. No GT is accepted. Inputs may be absent; this function never calls SAM.
    """
    import torch
    from scipy.ndimage import distance_transform_edt
    _sources(expected_sources)
    for frame in frames:_frame(frame)
    ids=tuple(r.identity for r in requests);keys=[identity_key(x) for x in ids]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate ROIidentity')
    if not isinstance(predictions,dict):raise ValueError('Predictions require identity-keyed mapping')
    bounds=[];indices=[]
    for r,key in zip(requests,keys):
        if type(r.image_index) is not int or not 0<=r.image_index<len(frames):raise ValueError('Image index out of range')
        f=frames[r.image_index]
        if key[0]!=f.image_id or r.source_pixel_space_id!=f.pixel_space_id:raise ValueError('ROI/native frame identity mismatch')
        if key[3]==f.pixel_space_id:raise ValueError('Output ROI pixel-space must differ from native source domain')
        b=np.asarray(r.bounds_native,float)
        if b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1] or not np.isclose(b[2]-b[0],b[3]-b[1],atol=1e-7,rtol=0):raise ValueError('Explicit finite squareROI required')
        bounds.append(b);indices.append(r.image_index)
    rgb,matrices=prepare_rois([f.pixels for f in frames],bounds,indices,roi_size=448,expansion=1.0,device='cpu')
    batch=len(requests);visible=torch.zeros(batch,1,448,448);distance=torch.zeros_like(visible);available=torch.zeros(batch,1,1,1);overlaps=torch.zeros(batch,1);valid=torch.zeros_like(visible)
    dropout=torch.zeros(batch,dtype=torch.bool) if dropout is None else dropout
    if not isinstance(dropout,torch.Tensor) or dropout.dtype!=torch.bool or tuple(dropout.shape)!=(batch,) or dropout.device.type!='cpu':raise ValueError('CPU bool dropout[B] required')
    dropout=dropout.clone()
    if (augmentation_policy is None)!=(augmentation_seeds is None):
        raise ValueError('Augmentation policy and per-request seeds must be supplied together')
    choices=None
    if augmentation_policy is not None:
        from .action4_sam_mask_augmentation import decision,apply
        if not isinstance(augmentation_seeds,(list,tuple)) or len(augmentation_seeds)!=batch:
            raise ValueError('Explicit augmentation seed per ROI required')
        choices=[decision(seed,key[0],key[1],augmentation_policy) for seed,key in zip(augmentation_seeds,keys)]
    reasons=[];dependencies=[];k112=[];k224=[]
    down112,_=DataAdapter.letterbox_affine((448,448),(112,112));down224,_=DataAdapter.letterbox_affine((448,448),(224,224))
    a448=np.asarray(matrices).reshape(batch,3,3);a112=down112@a448;a224=down224@a448
    for i,(r,key) in enumerate(zip(requests,keys)):
        frame=frames[r.image_index];native_id=NativeMaskIdentity(frame.image_id,r.identity.instance_id,r.source_pixel_space_id)
        native,overlap,reason,metadata=_prediction(predictions.get(native_key(native_id)),frame,native_id,expected_sources,allow_synthetic)
        grid=_native_nearest_grid(frame.pixels.shape[:2],a448[i])
        overlaps[i,0]=overlap;valid[i]=_sample_native_valid_on_grid(frame.pixels.shape[:2],grid)
        aug_log=None
        if choices is not None:
            aug_log=dict(choices[i],skip_reason='native_unavailable' if native is None else 'explicit_prediction_dropout' if bool(dropout[i]) else None)
        if native is not None and not bool(dropout[i]):
            sampled=_sample_native_nearest_on_grid(native,grid);m=sampled[0].numpy().astype(bool)
            had_pixels=bool(m.any())
            if choices is not None:
                m,aug_log=apply(m,valid[i,0].numpy().astype(bool),choices[i])
                if choices[i]['dropout'] and aug_log['applied']:dropout[i]=True
                sampled=torch.from_numpy(m.astype(np.float32))[None]
            if m.any() and not bool(dropout[i]):
                visible[i]=sampled;distance[i,0]=torch.from_numpy(np.clip((distance_transform_edt(m)-distance_transform_edt(~m))/16,-1,1).astype(np.float32));available[i]=1
            else:reason='empty_mask_after_augmentation' if choices is not None and had_pixels else 'empty_mask_after_roi'
        if bool(dropout[i]):reason='prediction_dropout'
        reasons.append(reason);dependencies.append(dict(expected_sources=expected_sources.copy(),cache_metadata=metadata,upstream_status=metadata['status'] if metadata is not None else 'cache_absent',upstream_reason=metadata['reason'] if metadata is not None else None,use_as_training_negative=False,native_shape=list(frame.pixels.shape[:2]),source_sha256=frame.source_sha256,variant_id=frame.variant_id,variant_sha256=frame.variant_sha256,source_pixel_space_id=frame.pixel_space_id,roi_pixel_space_id=r.identity.pixel_space_id,identity=list(key),native_cache_identity=list(native_key(native_id)),roi_bounds_native=list(map(float,r.bounds_native)),A448_sha256=hashlib.sha256(np.ascontiguousarray(a448[i]).tobytes()).hexdigest()))
        if aug_log is not None:
            aug_log['effective_dropout']=bool(dropout[i])
            dependencies[-1]['mask_augmentation']=aug_log
        if r.K_native is None:k112.append(None);k224.append(None)
        else:
            K=np.asarray(r.K_native,float)
            if K.shape!=(3,3) or not np.isfinite(K).all():raise ValueError('Invalid nativeK')
            k112.append(a112[i]@K);k224.append(a224[i]@K)
    aux=mask_aux_factory(visible=visible,distance=distance,available=available,candidate_overlap=overlaps,dropout=dropout.clone(),identities=ids)
    return SamRoiInputs(rgb,aux,a448,a112,a224,valid,tuple(reasons),tuple(dependencies),tuple(k112),tuple(k224))

def prepare_quality_targets(inputs,labels):
    """GT-only supervision constructor, separate from predicted input creation.

    labels map full identity tuple to {identity,native_mask,supervised}. Unknown
    labels produce NaN sentinels with supervisedFalse, never an inference mask.
    """
    import torch
    ids=inputs.mask_aux.identities;gt=torch.full_like(inputs.mask_aux.visible,float('nan'));supervised=torch.zeros(len(ids),dtype=torch.bool)
    for i,identity in enumerate(ids):
        label=labels.get(identity_key(identity))
        if label is None:continue
        if label.get('source_pixel_space_id')!=inputs.source_dependencies[i]['source_pixel_space_id']:raise ValueError('GT source pixel-space mismatch')
        if identity_key(label['identity'])!=identity_key(identity):raise ValueError('GT supervision identity mismatch')
        if type(label['supervised']) is not bool:raise ValueError('Explicit GT eligibility required')
        if not label['supervised']:continue
        native=label['native_mask']
        # Full native shape is bound in source metadata when present; labels must
        # also declare native_shape, including for unavailable prediction inputs.
        if list(label['native_shape'])!=inputs.source_dependencies[i]['native_shape'] or tuple(label['native_shape'])!=np.asarray(native).shape:raise ValueError('GT native shape mismatch')
        gt[i]=sample_native_nearest(native,inputs.A448[i]);supervised[i]=True
    return QualityTargets(gt,supervised,ids)
