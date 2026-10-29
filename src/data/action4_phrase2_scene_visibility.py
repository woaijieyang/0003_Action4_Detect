"""Actual-scene sparse ray attribution; measured mask areas remain authoritative."""
from collections import Counter
import numpy as np

def classify_source(name,own,other):
    if name is None:return "none_sampled"
    if name in own:return "own_wrist_forearm" if "forearm" in name else "own_mount"
    if name in other:return "other_actor"
    return "environment"

class SceneVisibility:
    def __init__(self,runtime,local_trees,resolution=24):
        self.runtime=runtime;self.local_trees=local_trees;self.resolution=resolution
    def measure(self,proposal):
        import bpy
        from mathutils import Vector
        from data import action4_global_render as renderer
        from data.action4_phrase2_geometry import projected
        r=self.runtime;scene=r.scene;contributors=set(renderer.integrated.contributing_objects(scene))
        excluded=[o for o in scene.objects if o.type in ["MESH","CURVE","SURFACE"] and o not in contributors]
        old={o.name:o.hide_viewport for o in excluded};result={}
        try:
            for o in excluded:o.hide_viewport=True
            bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
            for side,ident in [("left",1),("right",2)]:
                B=np.array(proposal["actor_pose_capture_by_side"][side]["brace_B_capture"]);C=np.array(proposal["actor_pose_capture_by_side"][side]["core_C_capture"]);cad=np.array(r.metadata["canonical_definition"]["corners_3d_m"])
                _,box,_=projected(cad,C,r.K)
                x=np.linspace(box[0]+.25,box[2]-.25,self.resolution);y=np.linspace(box[1]+.25,box[3]-.25,self.resolution);xx,yy=np.meshgrid(x,y)
                ray=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]@np.linalg.inv(r.K).T;ray/=np.linalg.norm(ray,axis=1)[:,None]
                inverse=np.linalg.inv(B);local=ray@inverse[:3,:3].T;origin=inverse[:3,3]
                target={o.name for o in r.targets()[ident]["parts"]};own=set(r.metadata["actor_groups"][side]["objects"]);counts=Counter();classes=Counter();total=visible=missing=0
                for capture_ray,local_ray in zip(ray,local):
                    hit,_,_,distance=self.local_trees[side].body.ray_cast(Vector(origin),Vector(local_ray))
                    if hit is None:continue
                    total+=1;world_ray=r.W[:3,:3]@capture_ray
                    found,location,_,_,obj,_=scene.ray_cast(deps,Vector(r.W[:3,3]),Vector(world_ray),distance=distance+1e-4)
                    if not found:visible+=1;missing+=1;continue
                    name=obj.name
                    if name in target:visible+=1;continue
                    actual=float(np.linalg.norm(np.array(location)-r.W[:3,3]))
                    if actual>=distance-1e-5:visible+=1;continue
                    counts[name]+=1
                    if name in own:
                        kind="own_wrist_forearm" if "forearm" in name else "own_mount"
                    elif any(name in group["objects"] for s,group in r.metadata["actor_groups"].items() if s!=side):kind="other_actor"
                    else:kind="environment"
                    classes[kind]+=1
                if total==0 or missing>max(4,.02*total):raise ValueError("Scene/body visibility binding mismatch")
                primary=counts.most_common(1)[0][0] if counts else None
                other=set().union(*(set(group["objects"]) for s,group in r.metadata["actor_groups"].items() if s!=side))
                aggregate_class=classes.most_common(1)[0][0] if classes else "none_sampled"
                primary_class=classify_source(primary,own,other)
                result[side]={"method":"actual scene sparse ray attribution; not pixel area","sampled_amodal_body_rays":total,"sampled_visible_body_rays":visible,"estimated_external_occlusion":1-visible/total,"occluder_object_counts":dict(counts),"occluder_class_counts":dict(classes),"primary_occluder_object":primary,"primary_source_class":primary_class,"aggregate_primary_source_class":aggregate_class,"primary_rule":"largest individual actual physical occluder; aggregate class shares reported separately"}
        finally:
            for name,value in old.items():scene.objects[name].hide_viewport=value
            bpy.context.view_layer.update()
        r.verify(scene);return result

def eligible_sources(sources,bins):
    for side,m in sources.items():
        o=m["estimated_external_occlusion"];kind=bins[side];lo,hi=[(0,.35),(.35,.5),(.5,.7)][kind]
        if o<max(0,lo-.04) or o>min(.74,hi+.04):return False,"scene_occlusion_estimate"
        if kind>0 and m["primary_source_class"]!="own_wrist_forearm":return False,"middle_heavy_requires_own_skin"
        counts=m["occluder_class_counts"];occluded=sum(counts.values())
        if occluded>=5 and counts.get("environment",0)+counts.get("other_actor",0)>.5*occluded:return False,"non_own_arm_primary_occlusion"
    return True,None
