"""Reusable local-BVH arm visibility and constant-size projection primitives."""
import math,numpy as np
from data import action4_parameter_sampling as previous

def axis_angle(axis,angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis);x,y,z=axis
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)

def align(a,b):
    a=np.asarray(a)/np.linalg.norm(a);b=np.asarray(b)/np.linalg.norm(b);cross=np.cross(a,b);norm=np.linalg.norm(cross)
    if norm<1e-10:return np.eye(3) if np.dot(a,b)>0 else axis_angle(np.cross(a,[1,0,0]) if abs(a[0])<.9 else np.cross(a,[0,1,0]),math.pi)
    return axis_angle(cross/norm,math.atan2(norm,np.dot(a,b)))

def rotation_for_arm(runtime,side,roll_deg,direction=(0,1,-.15)):
    base=np.array(runtime.metadata["actor_groups"][side]["base_brace_capture_m"])[:3,:3]
    axis=runtime.arm_axis_B[side];d=np.array(direction,dtype=float);d/=np.linalg.norm(d)
    return axis_angle(d,math.radians(roll_deg))@align(base@axis,d)@base

def pose_for_box(R_B,A,corners,K,size,uv,width_fraction):
    """Solve desired projected x extent/center without scaling CAD or changing K."""
    C=np.eye(4);C[:3,:3]=R_B@A[:3,:3];q=corners@C[:3,:3].T
    xlo=(uv[0]*size[0]-width_fraction*size[0]/2-K[0,2])/K[0,0]
    xhi=(uv[0]*size[0]+width_fraction*size[0]/2-K[0,2])/K[0,0]
    lo=np.max(xlo*q[:,2]-q[:,0]);hi=np.min(xhi*q[:,2]-q[:,0]);z=(lo-hi)/(xhi-xlo)
    if z<=-q[:,2].min()+.01:raise ValueError("Projection depth too close")
    tx=lo+xlo*z
    ty=(uv[1]*size[1]-K[1,2])/K[1,1]*z
    for _ in range(4):
        v=K[1,1]*(q[:,1]+ty)/(q[:,2]+z)+K[1,2];l=int(np.argmin(v));h=int(np.argmax(v))
        ty-=((v[l]+v[h])/2-uv[1]*size[1])/(K[1,1]*(1/(q[l,2]+z)+1/(q[h,2]+z))/2)
    C[:3,3]=[tx,ty,z];B=C@np.linalg.inv(A)
    return B,C

def projected(corners,C,K):
    cam=corners@C[:3,:3].T+C[:3,3];h=cam@K.T;uv=h[:,:2]/cam[:,2:]
    return uv,np.r_[uv.min(0),uv.max(0)],cam

class LocalArmVisibility:
    def __init__(self,runtime,side):
        from mathutils.bvhtree import BVHTree
        from data.action4_modular_assembly import world_geometry
        self.runtime=runtime;self.side=side;group=runtime.metadata["actor_groups"][side]
        inv=np.linalg.inv(runtime.W@np.array(group["base_brace_capture_m"]));verts=[];faces=[];labels=[];names=[];bodyverts=[];bodyfaces=[]
        hidden=set(runtime.bindings[side].get("always_hidden",[]))
        for name in group["objects"]:
            obj=runtime.scene.objects[name]
            if obj.type!="MESH" or obj.hide_render or name in hidden:continue
            v,f=world_geometry(obj);v=v@inv[:3,:3].T+inv[:3,3]
            isbody=name in [side+"_Action4_core",side+"_Action4_core_top_id_decal"]
            faces.extend((f+len(verts)).tolist());verts.extend(v.tolist());labels.extend([isbody]*len(f));names.extend([name]*len(f))
            if isbody:bodyfaces.extend((f+len(bodyverts)).tolist());bodyverts.extend(v.tolist())
        self.all=BVHTree.FromPolygons(verts,faces,all_triangles=True);self.body=BVHTree.FromPolygons(bodyverts,bodyfaces,all_triangles=True);self.labels=labels;self.names=names
    def estimate(self,B,C,resolution=28,return_sources=False):
        from mathutils import Vector
        runtime=self.runtime;cad=np.array(runtime.metadata["canonical_definition"]["corners_3d_m"])
        uv,box,_=projected(cad,C,runtime.K)
        x=np.linspace(box[0],box[2],resolution);y=np.linspace(box[1],box[3],resolution)
        xx,yy=np.meshgrid(x,y);pixels=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]
        inv=np.linalg.inv(B);origin=inv[:3,3];rays=pixels@np.linalg.inv(runtime.K).T@inv[:3,:3].T
        rays/=np.linalg.norm(rays,axis=1)[:,None]
        n=visible=0
        from collections import Counter
        counts=Counter()
        for ray in rays:
            loc,_,_,distance=self.body.ray_cast(Vector(origin),Vector(ray))
            if loc is None:continue
            n+=1
            hit,_,face,d=self.all.ray_cast(Vector(origin),Vector(ray))
            if hit is not None and self.labels[face] and abs(d-distance)<1e-4:visible+=1
            elif hit is not None:counts[self.names[face]]+=1
        value=None if not n else 1-visible/n
        if not return_sources:return value
        primary=counts.most_common(1)[0][0] if counts else None
        return {"occlusion_estimate":value,"primary_object":primary,"primary_class":"own_wrist_forearm" if primary and "forearm" in primary else "own_mount" if primary else "none_sampled"}


def upperarm_reenters(runtime,side,B,size):
    def inside(points):
        p=points@B[:3,:3].T+B[:3,3];positive=p[:,2]>.02
        if not positive.any():return False
        h=p[positive]@runtime.K.T;uv=h[:,:2]/h[:,2:]/size
        return bool(((uv>=0)&(uv<=1)).all(1).any())
    # Vertex-only image tests miss a large cap/face crossing the frustum when
    # all its vertices are outside or behind the near plane. A conservative
    # convex half-space test includes those faces without rebuilding geometry.
    p=runtime.upperarm_points_B[side]@B[:3,:3].T+B[:3,3]
    h=p@runtime.K.T;z=p[:,2]
    outside=[z-.02,h[:,0],size[0]*z-h[:,0],h[:,1],size[1]*z-h[:,1]]
    may_intersect=not any((plane<0).all() for plane in outside)
    return not inside(runtime.arm_junction_points_B[side]) and may_intersect

def junction_outside_frame(runtime,side,B,size):
    """Keep the coarse forearm/upper-arm junction entirely beyond a common edge."""
    points=runtime.arm_junction_points_B[side]@B[:3,:3].T+B[:3,3]
    start=B[:3,3]
    if start[2]<=.02:return False
    behind=points[:,2]<.02
    if behind.any():
        fraction=(start[2]-.02)/(start[2]-points[behind,2])
        points=points.copy();points[behind]=start+(points[behind]-start)*fraction[:,None]
    h=points@runtime.K.T;uv=h[:,:2]/h[:,2:]/size
    return bool((uv[:,0]<-.01).all() or (uv[:,0]>1.01).all() or (uv[:,1]<-.01).all() or (uv[:,1]>1.01).all())
