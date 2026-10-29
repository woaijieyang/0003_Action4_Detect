"""Explicit current-frame ownership for the native agimus camera bridge.

No renderer/model starts at import. Caller must use the CPU-only native launcher.
A camera stores a borrowed cv::Mat; this adapter owns contiguous BGR bytes until
all calls finish and retains them until the next frame replaces the camera image.
"""
import hashlib,json
import numpy as np


def pixel_digest(a):
    return hashlib.sha256(json.dumps(dict(shape=list(a.shape),dtype=str(a.dtype),strides=list(a.strides)),sort_keys=True).encode()+a.tobytes(order="C")).hexdigest()


class _FrameLease:
    def __init__(self,adapter):self.adapter=adapter
    def __enter__(self):return self.adapter
    def __exit__(self,kind,value,trace):
        try:self.adapter._finish()
        finally:self.adapter._active=False
        return False


class OwnedNativeCamera:
    def __init__(self,camera,*,width,height):
        self.camera=camera;self.width=width;self.height=height;self._owner=None;self._active=False;self.audit=[];self._entry=None
    def frame(self,pixels,*,frame_id,source_sha256):
        if self._active:raise RuntimeError("Previous native frame still active")
        if (not isinstance(pixels,np.ndarray) or pixels.dtype!=np.uint8 or pixels.shape!=(self.height,self.width,3)
            or not pixels.flags.c_contiguous or pixels.strides!=(self.width*3,3,1)):
            raise ValueError("Explicit contiguous native uint8 BGR frame required")
        if type(frame_id) is not int or not isinstance(source_sha256,str) or len(source_sha256)!=64:raise ValueError("Frame/source identity required")
        before=pixel_digest(pixels)
        owner=np.array(pixels,copy=True,order="C");owner.setflags(write=False)
        self.camera.image=owner
        self._owner=owner;self._active=True
        if not np.array_equal(owner,pixels):raise RuntimeError("Native owned copy differs from input")
        self._entry=dict(frame_id=frame_id,source_sha256=source_sha256,input_checksum=before,owner_checksum=before,shape=list(owner.shape),strides=list(owner.strides),pixel_order="BGR",owned_copy=True)
        return _FrameLease(self)
    def snapshot(self):
        if not self._active:raise RuntimeError("Native cv::Mat output must be copied inside active frame")
        return np.array(self.camera.image,copy=True,order="C")
    def pose(self,body):
        if not self._active:raise RuntimeError("Pose output must be copied inside active frame")
        return np.array(body.body2world_pose,dtype=np.float64,copy=True,order="C")
    def _finish(self):
        image=self.snapshot();after=pixel_digest(image)
        # Equality proves the second buffer has the SAME content; avoid hashing
        # two equal native full-resolution buffers again. Keep both invariants.
        same=np.array_equal(image,self._owner)
        owner_after=after if same else pixel_digest(self._owner)
        record=dict(self._entry,output_checksum=after,owner_checksum_after=owner_after,shape_after=list(image.shape),strides_after=list(image.strides))
        record["unchanged"]=after==owner_after==self._entry["input_checksum"]
        self.audit.append(record)
        if not record["unchanged"]:raise RuntimeError("Native frame bytes/shape/strides changed during consumption")


SE3_REPRESENTATION_POLICY = dict(schema="action4.native_float32_se3_representation.v1",
    source="native_float32_values",rotation_conversion="SVD_nearest_positive_determinant_rotation",
    max_orthogonality_error=32*float(np.finfo(np.float32).eps),
    max_determinant_error=32*float(np.finfo(np.float32).eps),
    max_singular_value_error=32*float(np.finfo(np.float32).eps),
    max_rotation_element_change=16*float(np.finfo(np.float32).eps),
    max_corner_projection_change_px=.01,translation_changed=False,native_state_changed=False,
    existing_geometry_gate_changed=False,accuracy_gain_claimed=False)


def native_float32_se3_for_projection(matrix,canonical8_m,K_native):
    """Bounded representation conversion only; NEVER write result into native state.

    Raw values and translation stay preserved. Gross nonrigidity/reflections,
    mixed precision, nonfinite values or a noncanonical bottom row fail closed.
    Geometry consumers must still run their original strict SO3/visibility gates.
    """
    raw=np.asarray(matrix)
    c=np.asarray(canonical8_m,dtype=np.float64);K=np.asarray(K_native,dtype=np.float64)
    if (raw.shape!=(4,4) or raw.dtype not in (np.dtype("float32"),np.dtype("float64"))
        or not np.isfinite(raw).all() or not np.array_equal(raw[3],[0,0,0,1])
        or c.shape!=(8,3) or not np.isfinite(c).all() or K.shape!=(3,3) or not np.isfinite(K).all()
        or not np.array_equal(K[2],[0,0,1]) or min(K[0,0],K[1,1])<=0):
        raise ValueError("Finite native4x4/unchanged bottom row/canonical8/K required")
    if not np.array_equal(raw.astype(np.float64),raw.astype(np.float32).astype(np.float64)):
        raise ValueError("Not exact native float32 values; do not silently quantize arbitrary pose")
    T=raw.astype(np.float64,copy=True);R=T[:3,:3];det=float(np.linalg.det(R));orth=float(abs(R.T@R-np.eye(3)).max());policy=SE3_REPRESENTATION_POLICY
    if det<=0 or abs(det-1)>policy["max_determinant_error"] or orth>policy["max_orthogonality_error"]:
        raise ValueError("Not machine-precision proper rigid rotation")
    u,s,vt=np.linalg.svd(R)
    if float(abs(s-1).max())>policy["max_singular_value_error"]:raise ValueError("Nonrigid singular values")
    converted=T.copy();proper=u@vt
    if np.linalg.det(proper)<=0:raise ValueError("Reflected nearest rotation is not repairable")
    delta=float(abs(proper-R).max())
    if delta>policy["max_rotation_element_change"]:raise ValueError("Representation change exceeds fixed float32 budget")
    converted[:3,:3]=proper
    def project(t):
        xyz=c@t[:3,:3].T+t[:3,3]
        if not (xyz[:,2]>1e-6).all():raise ValueError("Nonpositive canonical depth")
        h=xyz@K.T;return h[:,:2]/h[:,2:]
    raw_uv=project(T);converted_uv=project(converted);change=float(np.linalg.norm(raw_uv-converted_uv,axis=1).max())
    if change>policy["max_corner_projection_change_px"]:raise ValueError("Pixel change exceeds numerical representation budget")
    raw32=raw.astype(np.float32,copy=True);raw32.setflags(write=False);converted.setflags(write=False)
    return dict(raw_matrix=raw32,converted_matrix=converted,raw_corner_projection=raw_uv,
        converted_corner_projection=converted_uv,policy=dict(policy),raw_orthogonality_error=orth,
        raw_determinant_error=abs(det-1),rotation_element_change=delta,max_corner_projection_change_px=change,
        semantic_status="numerical_representation_only_not_pose_refinement",native_state_changed=False)
