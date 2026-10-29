"""Read-only light-to-skin ray attribution for the visible seam artifact."""
from pathlib import Path
import sys,json
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
import bpy
from mathutils import Vector
from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
r=CoverageArmRuntime('table_full_garment')
snapshot=json.loads((R/'outputs/coverage1000_v001/arm_repair_v001/visual_probe_v002/table_full_garment/00_C/FRAME_SNAPSHOT.json').read_text());r.apply(snapshot['proposal']);deps=bpy.context.evaluated_depsgraph_get()
lights=[o for o in r.scene.objects if o.type=='LIGHT'];rows=[]
for v in np.linspace(.20,.90,24):
    for u in np.linspace(.02,.98,36):
        ray=r.W[:3,:3]@np.linalg.inv(r.K)@np.array([u*3248,v*2464,1.]);ray/=np.linalg.norm(ray)
        found,location,normal,face,obj,_=r.scene.ray_cast(deps,Vector(r.W[:3,3]),Vector(ray))
        if not found or not obj.name.endswith('forearm_continuation'):continue
        point=np.array(location);N=np.array(normal);measure=[]
        for lamp in lights:
            delta=np.array(lamp.matrix_world.translation)-point;length=float(np.linalg.norm(delta));L=delta/length
            hit,loc,_,_,other,_=r.scene.ray_cast(deps,Vector(point+N*1e-5),Vector(L),distance=max(0,length-1e-5))
            emission_normal=-np.array(lamp.matrix_world)[:3,2];emission_normal/=np.linalg.norm(emission_normal)
            measure.append({'lamp':lamp.name,'N_dot_L':float(N@L),'emitter_front_cosine':float(emission_normal@(-L)),'caster':other.name if hit else None,'caster_distance_m':float(np.linalg.norm(np.array(loc)-point)) if hit else None})
        rows.append({'uv':[float(u),float(v)],'object':obj.name,'face':int(face),'light_rays':measure})
report={'lights':[{'name':o.name,'type':o.data.type,'energy':o.data.energy,'size':getattr(o.data,'size',None),'position':list(o.matrix_world.translation)} for o in lights],
    'rows':rows,'cycles_object_properties':[p.identifier for p in r.scene.objects['left_forearm_continuation'].cycles.bl_rna.properties]}
(R/'outputs/coverage1000_v001/arm_repair_v001/SHADOW_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
