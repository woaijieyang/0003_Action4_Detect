"""Same conservative contact/volume criteria as legacy, but fail fast on invalid candidates."""
import time,numpy as np
from data.action4_template_collision import (TemplateCollisionChecker,Geometry,mesh_signature,verify_evaluation_supported,
    transform,separated_axes,triangle_contact,point_in_closed,TOL)

def fast_pair(a,b,axes):
    from data.action4_modular_bilateral_scene import aabb_separation
    separation=aabb_separation(a.vertices,b.vertices)
    if max(separation["axis_gaps_m"])>TOL:return {"status":"CLEAR_AABB"}
    if separated_axes(a.vertices,b.vertices,axes):return {"status":"CLEAR_SEPARATING_AXIS"}
    pairs=a.bvh().overlap(b.bvh())
    for i,j in pairs:
        if triangle_contact(a.vertices[a.faces[i]],b.vertices[b.faces[j]]):
            return {"status":"PROHIBITED_SURFACE_CONTACT","first_triangle_pair":[int(i),int(j)]}
    details=[]
    for source,container in [(a,b),(b,a)]:
        if container.planar:details.append("ZERO_THICKNESS_SHEET");continue
        if not container.closed:details.append("UNKNOWN_OPEN_VOLUME");continue
        answer=[point_in_closed(source.vertices[i],container.vertices,container.faces) for i in source.representatives]
        if any(x is True for x in answer):return {"status":"PROHIBITED_CONTAINMENT"}
        details.append("UNKNOWN_RAY_PARITY" if any(x is None for x in answer) else "OUTSIDE_CLOSED_SOLID")
    return {"status":"UNKNOWN_CONTAINMENT" if any(x.startswith("UNKNOWN") for x in details) else "CLEAR_SURFACE_AND_CONTAINMENT","containment":details}

def joined_skin(a,b):
    vertices=np.concatenate([a.vertices,b.vertices]);faces=np.concatenate([a.faces,b.faces+len(a.vertices)])
    keys=np.round(vertices/1e-7).astype(np.int64)
    _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    faces=inverse[faces];vertices=vertices[first]
    # Weld only coincident source vertices; no surface point is moved.
    result=Geometry.create(vertices,faces)
    if not result.closed:raise ValueError("Joined wrist/forearm is not an oriented closed volume")
    return result

def capped_cable_proxy(base):
    if base.closed:return base
    counts={};orientation={}
    for face in base.faces:
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))));counts[key]=counts.get(key,0)+1;orientation[key]=(int(a),int(b))
    boundary=[orientation[e] for e,n in counts.items() if n==1]
    neighbors={}
    for a,b in boundary:neighbors.setdefault(a,[]).append(b);neighbors.setdefault(b,[]).append(a)
    if not boundary or any(len(v)!=2 for v in neighbors.values()):return base
    left=set(neighbors);vertices=base.vertices.tolist();faces=base.faces.tolist()
    while left:
        component={left.pop()};todo=list(component)
        while todo:
            for v in neighbors[todo.pop()]:
                if v in left:left.remove(v);component.add(v);todo.append(v)
        points=base.vertices[sorted(component)];singular=np.linalg.svd(points-points.mean(0),compute_uv=False)
        if len(component)>128 or np.max(np.ptp(points,axis=0))>.03 or singular[-1]>1e-7*len(points)**.5:return base
        center=len(vertices);vertices.append(points.mean(0).tolist())
        for a,b in boundary:
            if a in component:faces.append([b,a,center])
    result=Geometry.create(vertices,faces)
    return result if result.closed else base

def enclosing_box_proxy(base,matrix):
    """Closed local box encloses every rendered mesh vertex; conservative only."""
    matrix=np.asarray(matrix);local=transform(np.linalg.inv(matrix),base.vertices);low=local.min(0);high=local.max(0)
    vertices=np.array([[x,y,z] for z in [low[2],high[2]] for y in [low[1],high[1]] for x in [low[0],high[0]]])[[0,1,3,2,4,5,7,6]]
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    result=Geometry.create(transform(matrix,vertices),faces)
    if not result.closed or not ((local>=low-1e-9)&(local<=high+1e-9)).all():raise ValueError("Conservative body proxy enclosure failed")
    return result

class Phrase2CollisionChecker(TemplateCollisionChecker):
    def __init__(self,runtime):
        super().__init__(runtime)
        self.skin_proxies={};self.closed_components={};self.proxy_metadata={}
        for name,base in self.cache.items():
            if "cable" in name or name in ["mount_left_link","mount_right_link"] or runtime.scene.objects[name].type=="CURVE":
                closed=capped_cable_proxy(base)
                if closed.closed and not base.closed:
                    self.closed_components[name]=closed;self.proxy_metadata[name]="closed proxy of small planar endpoint rings; rendered mesh unchanged"
            if name.endswith("_Action4_core") and not base.closed:
                self.closed_components[name]=enclosing_box_proxy(base,runtime.world[name]);self.proxy_metadata[name]="closed local OBB strictly encloses all rendered core vertices; no geometry modification"
        for side in ["left","right"]:
            members=[side+"_forearm_natural_hand",side+"_forearm_continuation"]
            geometry=joined_skin(self.cache[members[0]],self.cache[members[1]])
            self.skin_proxies[side]=(members,geometry)
    def _check(self,report):
        from data.action4_modular_assembly import world_geometry
        r=self.runtime;r.verify(r.scene);geometries={};rotations={};deltas={}
        report["conservative_collision_proxies"]=self.proxy_metadata
        for name in self.excluded_baseline_hidden:
            if not r.scene.objects[name].hide_render:raise ValueError("Hidden baseline geometry became visible")
        for name,base in self.cache.items():
            obj=r.scene.objects[name];verify_evaluation_supported(obj)
            if mesh_signature(obj)!=self.signatures[name]:raise ValueError("Source geometry changed")
            delta=np.array(obj.matrix_world)@np.linalg.inv(r.world[name]);rotation=delta[:3,:3]
            if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-5,rtol=0) or not np.isclose(np.linalg.det(rotation),1.,atol=2e-5):raise ValueError("Nonrigid delta")
            if self.signatures[name] is None:
                v,f=world_geometry(obj)
                if not np.array_equal(f,base.faces) or not np.allclose(v,transform(delta,base.vertices),atol=3e-6,rtol=0):raise ValueError("Evaluated geometry changed")
            group=self.groups[name]
            if group=="static" and not np.allclose(delta,np.eye(4),atol=3e-6,rtol=0):raise ValueError("Static moved after init")
            if group in deltas and not np.allclose(deltas[group],delta,atol=5e-6,rtol=0):raise ValueError("Relative module transform changed")
            deltas[group]=delta
            if obj.hide_render:continue
            shape=self.closed_components.get(name,base)
            geometries[name]=shape if group=="static" else shape.moved(delta);rotations[name]=rotation
        groups=dict(self.groups)
        for side,(members,base) in self.skin_proxies.items():
            name=side+"_closed_skin_collision_proxy"
            delta=deltas[side];geometries[name]=base.moved(delta);rotations[name]=delta[:3,:3];groups[name]=side
            for member in members:geometries.pop(member,None);rotations.pop(member,None)
        names=sorted(geometries);checked=reused=0
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if groups[a]==groups[b]:reused+=1;continue
                checked+=1;axes=list(rotations[a].T)+list(rotations[b].T)
                axes.extend(np.cross(x,y) for x in rotations[a].T for y in rotations[b].T)
                result=fast_pair(geometries[a],geometries[b],axes);status=result["status"]
                report["pair_status_counts"][status]=report["pair_status_counts"].get(status,0)+1
                if status.startswith("PROHIBITED") or status.startswith("UNKNOWN"):
                    report["prohibited" if status.startswith("PROHIBITED") else "unknown"].append({"objects":[a,b],**result})
                    report.update(checked_changed_pairs=checked,baseline_invariant_pairs_reused=reused,early_rejection=True);return
        report.update(checked_changed_pairs=checked,baseline_invariant_pairs_reused=reused,active_geometry_objects=len(geometries),finite_geometry=True)
