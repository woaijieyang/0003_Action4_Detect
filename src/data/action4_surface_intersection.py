"""Double precision ray/known-triangle intersection after BVH face lookup."""
import numpy as np

def exact_surface_hits(native_pixels,K,object_to_capture,triangles):
    uv=np.asarray(native_pixels,dtype=np.float64);K=np.asarray(K,dtype=np.float64);T=np.asarray(object_to_capture,dtype=np.float64);tri=np.asarray(triangles,dtype=np.float64)
    if uv.ndim!=2 or uv.shape[1]!=2 or tri.shape!=(len(uv),3,3):raise ValueError('Ray/triangle shape mismatch')
    if not all(np.isfinite(x).all() for x in [uv,K,T,tri]):raise ValueError('Nonfinite ray inputs')
    inverse=np.linalg.inv(T);origin=inverse[:3,3];rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T@inverse[:3,:3].T
    a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];normal=np.cross(a,b);norm2=(normal*normal).sum(1);den=(rays*normal).sum(1)
    reliable=(norm2>0)&(np.abs(den)>np.sqrt(norm2)*1e-12)
    distance=((tri[:,0]-origin)*normal).sum(1)/np.where(reliable,den,1)
    xyz=origin+rays*distance[:,None];q=xyz-tri[:,0];safe=np.where(norm2>0,norm2,1)
    u=(np.cross(q,b)*normal).sum(1)/safe;v=(np.cross(a,q)*normal).sum(1)/safe;bary=np.c_[1-u-v,u,v]
    valid=reliable&(distance>0)&np.isfinite(xyz).all(1)&(bary.min(1)>=-1e-8)&(bary.max(1)<=1+1e-8)
    return {'xyz_object_m':xyz,'valid':valid,'barycentric':bary,'nonparallel':reliable,'positive_ray_distance':distance>0}
