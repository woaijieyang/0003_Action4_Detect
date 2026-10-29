"""CPU physical plane candidates, never exterior/semantic FaceID admission."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class PatchThresholds:
    max_axis_angle_degrees: float
    depth_bin_mm: float
    max_plane_residual_mm: float
    max_boundary_inset_fraction: float
    minimum_area_mm2: float

    def __post_init__(self):
        values=asdict(self)
        if any(isinstance(v,bool) or not np.isfinite(v) for v in values.values()):
            raise ValueError("Finite explicit thresholds required")
        if not (0 <= self.max_axis_angle_degrees < 45 and self.depth_bin_mm > 0
                and self.max_plane_residual_mm >= 0 and self.minimum_area_mm2 >= 0
                and 0 <= self.max_boundary_inset_fraction < .5):
            raise ValueError("Invalid diagnostic thresholds")


def build_patch_candidates(vertices_mm, faces, first_hit_by_view, *, mesh_sha256, thresholds):
    """Return JSON data with every original triangle retained in mapping.

    View keys are integer logical axis directions 0..5. Their triangle sets must
    originate in authenticated first-hit evidence. This function does not claim
    it can distinguish a hole-visible internal surface from an exterior patch.
    """
    if not isinstance(thresholds,PatchThresholds):
        raise TypeError("Explicit PatchThresholds required")
    if not isinstance(mesh_sha256,str) or len(mesh_sha256)!=64 or any(c not in "0123456789abcdef" for c in mesh_sha256):
        raise ValueError("Mesh SHA256 required")
    v=np.asarray(vertices_mm,dtype=float);f=np.asarray(faces)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():
        raise ValueError("Finite nonempty Nx3 vertices required")
    if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in "iu" or not len(f) or f.min()<0 or f.max()>=len(v):
        raise ValueError("Original nonempty triangle index array required")
    if set(first_hit_by_view)!=set(range(6)) or any(type(k) is not int for k in first_hit_by_view):
        raise ValueError("Explicit six first-hit views required")
    observed=[set() for _ in f]
    for view,indices in first_hit_by_view.items():
        for index in indices:
            if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<len(f):
                raise ValueError("Invalid first-hit triangle index")
            observed[int(index)].add(view)
    tri=v[f];center=tri.mean(1)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1);area=length/2
    normal=np.divide(cross,length[:,None],out=np.zeros_like(cross),where=length[:,None]>0)
    axes=np.argmax(np.abs(normal),axis=1)
    aligned=np.max(np.abs(normal),axis=1)>=np.cos(np.deg2rad(thresholds.max_axis_angle_degrees))
    usable=aligned & (length>0)
    keys=[(int(axes[i]),int(np.floor(center[i,axes[i]]/thresholds.depth_bin_mm+.5))) if usable[i] else None for i in range(len(f))]
    parent=np.arange(len(f))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    edges={}
    for i,face in enumerate(f):
        if keys[i] is None:continue
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            key=(min(int(a),int(b)),max(int(a),int(b)),keys[i])
            if key in edges:union(i,edges[key])
            else:edges[key]=i
    groups={}
    for i in range(len(f)):
        if usable[i]:groups.setdefault(find(i),[]).append(i)
    bounds=np.array([v.min(0),v.max(0)])
    mapping=[{"triangle_index":i,"physical_patch_id":None,
              "logical_face_candidates":[],"first_hit_views":sorted(observed[i]),
              "sampling_status":"sampled_first_hit" if observed[i] else "unhit_unknown",
              "reason":"degenerate_triangle" if length[i]==0 else "non_near_axis_triangle"} for i in range(len(f))]
    patches=[]
    for indices in groups.values():
        idx=np.array(indices);points=tri[idx].reshape(-1,3)
        weights=np.repeat(area[idx]/3,3);origin=np.average(points,axis=0,weights=weights)
        _,_,vt=np.linalg.svd((points-origin)*np.sqrt(weights[:,None]),full_matrices=False)
        n=vt[-1];dominant=int(np.argmax(np.abs(n)))
        if n[dominant]<0:n=-n
        distance=(points-origin)@n
        maximum=float(np.max(np.abs(distance)));rms=float(np.sqrt(np.average(distance**2,weights=weights)))
        valid=maximum<=thresholds.max_plane_residual_mm and area[idx].sum()>=thresholds.minimum_area_mm2
        pid="patch_%06d"%len(patches)
        views=sorted(set().union(*(observed[i] for i in indices)))
        candidates=[];excluded={}
        for view in views:
            axis,side=divmod(view,2);sign=-1 if side==0 else 1
            extent=bounds[1,axis]-bounds[0,axis]
            inside=sign*(bounds[side,axis]-origin[axis])
            same_half=bool(np.all(sign*(points[:,axis]-bounds[:,axis].mean())>=0))
            if not valid:excluded[str(view)]="plane_residual_or_area_gate_failed"
            elif axis!=dominant:excluded[str(view)]="view_not_parallel_to_patch_plane_normal"
            elif not same_half:excluded[str(view)]="opposite_or_crossing_halfspace_not_exterior"
            elif extent<=0 or inside/extent>thresholds.max_boundary_inset_fraction:
                excluded[str(view)]="deep_inset_or_zero_extent_not_exterior"
            else:candidates.append(view)
        reasons=[]
        if not valid:reasons.append("plane_residual_or_area_gate_failed")
        if not views:reasons.append("unhit_unknown_not_background")
        reasons.append("hole_visible_internal_surface_not_ruled_out")
        winding=normal[idx]@n
        patch={"physical_patch_id":pid,"triangle_indices":indices,"logical_face_candidates":candidates,
               "observed_first_hit_views":views,"excluded_view_reasons":excluded,
               "plane_equation_n_dot_x_plus_d_equals_0":{"normal_object":n.tolist(),"d_mm":float(-n@origin)},
               "centroid_object_mm":origin.tolist(),"bounds_mm":[points.min(0).tolist(),points.max(0).tolist()],
               "plane_max_residual_mm":maximum,"plane_rms_residual_mm":rms,"area_mm2":float(area[idx].sum()),
               "sampled_triangle_count":sum(bool(observed[i]) for i in indices),
               "unhit_triangle_count":sum(not observed[i] for i in indices),
               "winding_positive_count":int(np.count_nonzero(winding>=0)),
               "winding_negative_count":int(np.count_nonzero(winding<0)),
               "geometric_plane_candidate":bool(valid),"exterior_verified":False,"semantic_gt":False,
               "unknown_or_exclusion_reasons":reasons}
        patches.append(patch)
        for i in indices:
            mapping[i].update(physical_patch_id=pid,logical_face_candidates=list(candidates),
                              reason="candidate_only_not_semantic_gt" if candidates else "patch_unknown_or_excluded")
    return {"schema":"action4.physical_patch_candidates.v1","status":"CANDIDATES_NOT_REGISTERED_NOT_SEMANTIC_GT",
            "mesh_sha256":mesh_sha256,"triangle_index_domain":"unchanged input faces row order",
            "thresholds":asdict(thresholds),"thresholds_calibrated":False,
            "normal_convention":"SVD canonical largest component positive; never assumed outward",
            "connectivity":"shared vertex-index edges; no vertex welding; fixed depth bins may split a plane",
            "visibility":"finite first-hit triangle evidence only; no area visibility proof",
            "patches":patches,"triangle_mapping":mapping,"training_labels_modified":False}
