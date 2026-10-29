"""View-dependent candidate curves from fixed CAD mesh, never image observations."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MeshEdgePolicy:
    spacing_m:float=.0008
    crease_angle_deg:float=25.
    occlusion_epsilon_m:float=.00002
    ray_chunk:int=8
    def __post_init__(self):
        if (not np.isfinite([self.spacing_m,self.crease_angle_deg,self.occlusion_epsilon_m]).all()
            or self.spacing_m<=0 or not 0<self.crease_angle_deg<180 or self.occlusion_epsilon_m<=0
            or type(self.ray_chunk) is not int or not 1<=self.ray_chunk<=32):
            raise ValueError("Explicit finite mesh curve policy required")

def ray_visible(origin,points,vertices,faces,*,epsilon_m,chunk):
    """Finite ray segments; any nearer triangle blocks. No backface culling."""
    tri=vertices[faces];v0=tri[:,0];e1=tri[:,1]-v0;e2=tri[:,2]-v0
    s=origin-v0;q=np.cross(s,e1)
    numerator=np.einsum("ij,ij->i",e2,q)
    visible=[]
    for begin in range(0,len(points),chunk):
        vector=points[begin:begin+chunk]-origin
        length=np.linalg.norm(vector,axis=1)
        if (length<=epsilon_m).any():raise ValueError("Camera is on candidate surface")
        directions=vector/length[:,None]
        h=np.cross(directions[:,None,:],e2[None,:,:])
        det=np.einsum("fj,nfj->nf",e1,h)
        valid=np.abs(det)>1e-14
        inv=np.zeros_like(det);inv[valid]=1/det[valid]
        u=np.einsum("fj,nfj->nf",s,h)*inv
        v=np.einsum("nj,fj->nf",directions,q)*inv
        depth=numerator[None,:]*inv
        hits=valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(depth>epsilon_m)&(depth<length[:,None]-epsilon_m)
        visible.extend((~hits.any(1)).tolist())
    return np.asarray(visible,bool)

def candidate_curves(vertices_m,faces,component_triangle_ids,R,t_m,*,policy):
    if not isinstance(policy,MeshEdgePolicy):raise TypeError("Typed policy required")
    v=np.asarray(vertices_m,float);f=np.asarray(faces);ids=np.asarray(component_triangle_ids)
    R=np.asarray(R,float);t=np.asarray(t_m,float)
    if (v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3
        or f.dtype.kind not in "iu" or not len(f) or f.min()<0 or f.max()>=len(v)
        or ids.ndim!=1 or ids.dtype.kind not in "iu" or len(set(ids.tolist()))!=len(ids) or not len(ids)
        or ids.min()<0 or ids.max()>=len(f)):
        raise ValueError("Finite fixed metre vertices and unique valid mesh triangle IDs required")
    if (R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1,atol=1e-6)):
        raise ValueError("Proper finite object-to-capture pose required")
    camera=-R.T@t;tri=v[f[ids]]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normals,axis=1)
    if (length<=1e-15).any():raise ValueError("Degenerate selected mesh face")
    normals/=length[:,None];centers=tri.mean(1)
    front=np.einsum("ij,ij->i",normals,camera-centers)>0
    edges={}
    for local,face in enumerate(f[ids]):
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))));edges.setdefault(key,[]).append(local)
    records=[];all_points=[];lengths=[]
    for edge,adjacent in sorted(edges.items()):
        if len(adjacent)>2:
            records.append(dict(vertex_ids=list(edge),status="nonmanifold_unknown"));continue
        kind=None
        if len(adjacent)==1 and front[adjacent[0]]:kind="open_boundary_candidate"
        if len(adjacent)==2:
            a,b=adjacent
            if front[a]!=front[b]:kind="view_silhouette_candidate"
            elif front[a] and np.dot(normals[a],normals[b])<np.cos(np.deg2rad(policy.crease_angle_deg)):
                kind="visible_mesh_crease_candidate"
        if kind is None:continue
        a,b=v[list(edge)];n=max(3,int(np.ceil(np.linalg.norm(b-a)/policy.spacing_m))+1)
        # Mid-segment samples avoid shared endpoint/ray ambiguities.
        points=a+(b-a)*((np.arange(n)+.5)/n)[:,None]
        records.append(dict(vertex_ids=list(edge),adjacent_triangle_ids=ids[adjacent].tolist(),status=kind,
                            sample_offset=len(all_points),sample_count=n))
        all_points.extend(points);lengths.append(n)
    if not all_points:
        return dict(status="no_view_dependent_candidates",records=records,curves_m=[],observed=False)
    pts=np.asarray(all_points);depth=(pts@R.T+t)[:,2]
    visible=ray_visible(camera,pts,v,f,epsilon_m=policy.occlusion_epsilon_m,chunk=policy.ray_chunk)&(depth>1e-6)
    curves=[];accepted_records=[]
    for rec in records:
        if "sample_offset" not in rec:continue
        start=rec["sample_offset"];n=rec["sample_count"];mask=visible[start:start+n]
        rec["visible_samples"]=int(mask.sum());rec["occluded_or_nonpositive_samples"]=int((~mask).sum())
        # Separate visible runs; do not bridge occluded mesh segments.
        run=[]
        for j,valid in enumerate(mask):
            if valid:run.append(pts[start+j].tolist())
            elif run:
                if len(run)>=3:curves.append(run);accepted_records.append(rec["vertex_ids"])
                run=[]
        if len(run)>=3:curves.append(run);accepted_records.append(rec["vertex_ids"])
    return dict(status="CAD_VIEW_CURVES_NOT_IMAGE_MEASUREMENTS",records=records,curves_m=curves,
        curve_vertex_ids=accepted_records,all_candidate_samples=len(pts),visible_samples=int(visible.sum()),
        observed=False,semantic_correspondence_verified=False,raw_mask_not_used=True,
        units="m",projection_pose_is_prior=True)
