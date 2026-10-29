"""Opt-in process-local bounded pure-input LRU. Not wired into inference.

External current-source validation MUST execute on every access, including hits.
Verified native objects authenticate their bytes; they do not prove that an old
object still corresponds to the caller's current image/file. No A0 state belongs
in this cache. Budget covers owned entries (including conservative metadata
charges), not caller inputs, returned clones, allocator arenas or process RSS.
"""
from collections import OrderedDict
from dataclasses import fields,is_dataclass
import hashlib,json,os,sys,threading
import numpy as np
from .action4_sam_roi_inputs import (SamRoiInputs,VerifiedImageFrame,VerifiedNativeSamPrediction,
    NativeMaskIdentity,native_key,identity_key,_frame,_prediction,_sources,_digest)

SAMPLING_FIELDS={"rgb_implementation_sha256","mask_implementation_sha256","aux_implementation_sha256",
                 "backend","rgb_dtype","grid_dtype","affine_dtype","torch_version","numpy_version","scipy_version"}
AUX_FIELDS={"visible","distance","available","candidate_overlap","dropout","identities","protocol_id"}


def _clone(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().clone(memory_format=torch.contiguous_format)
    if isinstance(value,np.ndarray):return value.copy(order="C")
    if is_dataclass(value):return type(value)(**{f.name:_clone(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,dict):return {k:_clone(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_clone(v) for v in value)
    if isinstance(value,list):return [_clone(v) for v in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise TypeError("Unsupported cache payload object")


def _charge(value):
    """Conservative owned-byte accounting; shared metadata is overcounted."""
    import torch
    if isinstance(value,torch.Tensor):return sys.getsizeof(value)+value.numel()*value.element_size()
    if isinstance(value,np.ndarray):return sys.getsizeof(value)+value.nbytes
    if is_dataclass(value):return sys.getsizeof(value)+sys.getsizeof(getattr(value,"__dict__",{}))+sum(_charge(getattr(value,f.name)) for f in fields(value))
    if isinstance(value,dict):return sys.getsizeof(value)+sum(_charge(k)+_charge(v) for k,v in value.items())
    if isinstance(value,(tuple,list)):return sys.getsizeof(value)+sum(_charge(v) for v in value)
    return sys.getsizeof(value)


def _make_key(frames,requests,predictions,expected_sources,*,frozen_b0_predictions_sha256,sampling,mode,roi_expansion):
    if mode!="frozen_original_image_eval" or roi_expansion!=1.5:raise ValueError("Only frozen original-image ROI1.5 evaluation is cacheable")
    _sources(expected_sources);_digest(frozen_b0_predictions_sha256)
    if set(sampling)!=SAMPLING_FIELDS or sampling["backend"]!="cpu":raise ValueError("Exact CPU sampling implementation/precision identity required")
    for k in SAMPLING_FIELDS:
        if k.endswith("sha256"):_digest(sampling[k])
        elif not isinstance(sampling[k],str) or not sampling[k]:raise ValueError("Sampling identity missing")
    if (sampling["rgb_dtype"],sampling["grid_dtype"],sampling["affine_dtype"])!=("float32","float32","float64"):
        raise ValueError("Registered prepared-input precision required")
    for f in frames:
        if not isinstance(f,VerifiedImageFrame):raise ValueError("Caller must supply current verified immutable frame")
        _frame(f)
        if f.variant_id not in ("original","native_original") or f.pixel_space_id.startswith("raw_video"):raise ValueError("Augmented RGB cannot enter frozen cache")
    ordered=[];seen=set()
    for r in requests:
        key=identity_key(r.identity)
        if key in seen:raise ValueError("Duplicate ROI identity")
        seen.add(key)
        if type(r.image_index) is not int or not 0<=r.image_index<len(frames):raise ValueError("Image index differs")
        f=frames[r.image_index]
        if key[0]!=f.image_id or r.source_pixel_space_id!=f.pixel_space_id or key[3]==f.pixel_space_id:raise ValueError("Native/ROI identity differs")
        bounds=np.asarray(r.bounds_native,float);K=np.asarray(r.K_native,float)
        if (bounds.shape!=(4,) or not np.isfinite(bounds).all() or bounds[2]<=bounds[0]
            or not np.isclose(bounds[2]-bounds[0],bounds[3]-bounds[1],rtol=0,atol=1e-7)
            or K.shape!=(3,3) or not np.isfinite(K).all()):raise ValueError("Actual square ROI/native K required")
        ni=NativeMaskIdentity(f.image_id,key[1],f.pixel_space_id);pred=predictions.get(native_key(ni))
        if pred is not None and not isinstance(pred,VerifiedNativeSamPrediction):raise ValueError("Current verified immutable SAM prediction required")
        _,_,_,metadata=_prediction(pred,f,ni,expected_sources,False)
        ordered.append(dict(identity=key,image_index=r.image_index,bounds=bounds.tolist(),K=K.tolist(),
            frame=dict(image_id=f.image_id,source=f.source_sha256,pixels=f.variant_sha256,variant=f.variant_id,
                       native_domain=f.pixel_space_id,shape=list(f.pixels.shape),receipt=f.validation_receipt),
            sam_metadata=metadata))
    if not ordered:raise ValueError("No ROI entries to cache")
    doc=dict(schema="action4.frozen_prepared_roi_cache.v1",ordered=ordered,sources=expected_sources,
             frozen_b0_predictions_sha256=frozen_b0_predictions_sha256,sampling=sampling,mode=mode,roi_expansion=roi_expansion)
    return hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _validate_payload(value,requests,frames,predictions,expected_sources):
    import torch
    if type(value) is not SamRoiInputs:raise ValueError("Only SamRoiInputs pure inputs may be cached")
    aux=value.mask_aux;n=len(requests)
    if not is_dataclass(aux) or {f.name for f in fields(aux)}!=AUX_FIELDS:raise ValueError("MaskAux pure-input schema required")
    if tuple(identity_key(x) for x in aux.identities)!=tuple(identity_key(r.identity) for r in requests):raise ValueError("Prepared instance order differs")
    shapes={"rgb":(n,3,448,448),"valid_image_support":(n,1,448,448)}
    tensors=[(k,getattr(value,k),shape,torch.float32) for k,shape in shapes.items()]
    tensors += [(k,getattr(aux,k),shape,dtype) for k,shape,dtype in [
        ("visible",(n,1,448,448),torch.float32),("distance",(n,1,448,448),torch.float32),
        ("available",(n,1,1,1),torch.float32),("candidate_overlap",(n,1),torch.float32),("dropout",(n,),torch.bool)]]
    for name,t,shape,dtype in tensors:
        if not isinstance(t,torch.Tensor) or tuple(t.shape)!=shape or t.device.type!="cpu" or t.dtype!=dtype or t.requires_grad or not torch.isfinite(t).all():raise ValueError("Invalid CPU prepared tensor: "+name)
    if aux.dropout.any():raise ValueError("Training/dropout inputs cannot be cached")
    for name in ("A448","A112","A224"):
        a=getattr(value,name)
        if not isinstance(a,np.ndarray) or a.shape!=(n,3,3) or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError("Invalid affine input")
    if len(value.reasons)!=n or len(value.source_dependencies)!=n or len(value.K112)!=n or len(value.K224)!=n:raise ValueError("Prepared batch dimensions differ")
    dep_fields={"expected_sources","cache_metadata","upstream_status","upstream_reason","use_as_training_negative","native_shape","source_sha256","variant_id","variant_sha256","source_pixel_space_id","roi_pixel_space_id","identity","native_cache_identity","roi_bounds_native","A448_sha256"}
    for i,(dep,r) in enumerate(zip(value.source_dependencies,requests)):
        if set(dep)!=dep_fields or dep.get("use_as_training_negative") is not False:raise ValueError("Only registered prediction-input dependencies are cacheable")
        f=frames[r.image_index];ni=NativeMaskIdentity(f.image_id,r.identity.instance_id,f.pixel_space_id)
        _,_,_,metadata=_prediction(predictions.get(native_key(ni)),f,ni,expected_sources,False)
        if (dep["expected_sources"]!=expected_sources or dep["cache_metadata"]!=metadata
            or dep["identity"]!=list(identity_key(r.identity)) or dep["native_cache_identity"]!=list(native_key(ni))
            or dep["roi_bounds_native"]!=list(map(float,r.bounds_native)) or dep["source_sha256"]!=f.source_sha256
            or dep["variant_sha256"]!=f.variant_sha256 or dep["variant_id"]!=f.variant_id
            or dep["source_pixel_space_id"]!=f.pixel_space_id
            or dep["A448_sha256"]!=hashlib.sha256(np.ascontiguousarray(value.A448[i]).tobytes()).hexdigest()):
            raise ValueError("Prepared payload source/ROI identity differs from requested key")
    for name in ("K112","K224"):
        for a in getattr(value,name):
            if not isinstance(a,np.ndarray) or a.shape!=(3,3) or not np.isfinite(a).all():raise ValueError("Prepared native camera input required")


class FrozenPreparedRoiCache:
    """One worker, across checkpoints; never pickle or share across processes."""
    def __init__(self,max_bytes=8*1024**3):
        if type(max_bytes) is not int or max_bytes<0:raise ValueError("Explicit nonnegative byte budget required")
        self.max_bytes=max_bytes;self._entries=OrderedDict();self._bytes=0;self._pid=os.getpid();self._lock=threading.RLock()
        self.hits=0;self.misses=0;self.evictions=0;self.oversize=0
    @property
    def resident_bytes(self):return self._bytes
    def __len__(self):return len(self._entries)
    def __reduce_ex__(self,protocol):raise TypeError("Prepared cache is process-local")
    def clear(self):
        with self._lock:self._entries.clear();self._bytes=0
    def get_or_prepare(self,*,frames,requests,predictions,expected_sources,frozen_b0_predictions_sha256,sampling,
                       validate_current,prepare,mode="frozen_original_image_eval",roi_expansion=1.5):
        if os.getpid()!=self._pid:raise ValueError("Inherited/forked cache cannot be reused")
        if not callable(validate_current) or not callable(prepare):raise TypeError("Current identity validator and pure preparation callback required")
        # Intentionally before key construction or hit lookup. A caller's old
        # key cannot bypass external source/current-frame checks.
        if validate_current() is not True:raise ValueError("Current input identity verification did not pass")
        key=_make_key(frames,requests,predictions,expected_sources,frozen_b0_predictions_sha256=frozen_b0_predictions_sha256,sampling=sampling,mode=mode,roi_expansion=roi_expansion)
        with self._lock:
            if key in self._entries:
                value,charge=self._entries.pop(key);self._entries[key]=(value,charge);self.hits+=1
                return _clone(value)
            self.misses+=1
            value=prepare();_validate_payload(value,requests,frames,predictions,expected_sources)
            # 1024 bytes/entry conservatively covers map node, digest key and
            # tuple overhead; arrays count data twice rather than undercount.
            charge=_charge(value)+1024
            if charge>self.max_bytes:
                self.oversize+=1
                return _clone(value)
            while self._bytes+charge>self.max_bytes:
                _,(_,old)=self._entries.popitem(last=False);self._bytes-=old;self.evictions+=1
            stored=_clone(value);self._entries[key]=(stored,charge);self._bytes+=charge
            return _clone(stored)
