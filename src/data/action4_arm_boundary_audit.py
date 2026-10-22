"""Read-only Blender mesh boundary diagnostics; no asset mutation or rendering."""
from pathlib import Path
import json,sys
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
import numpy as np
from data.action4_template_runtime import TemplateRuntime,read


def run():
    out=R/'outputs/action4_arm_boundary_audit_v001';out.mkdir(exist_ok=True)
    cases=read('configs/data/action4_counterfactual_diagnostic_v002.json')['cases'];rows=[]
    for case in cases:
        runtime=TemplateRuntime(case['template']);proposal=read(case['parameters_ref']['path']);runtime.apply(proposal)
        inverse=np.linalg.inv(np.array(runtime.metadata['W_from_capture']));K=np.array(runtime.metadata['K_native'])
        for obj in runtime.scene.objects:
            if obj.type!='MESH' or 'forearm' not in obj.name:continue
            mesh=obj.data;verts=np.empty((len(mesh.vertices),3));mesh.vertices.foreach_get('co',verts.ravel())
            counts={}
            for poly in mesh.polygons:
                vv=list(poly.vertices)
                for a,b in zip(vv,vv[1:]+vv[:1]):
                    key=tuple(sorted((a,b)));counts[key]=counts.get(key,0)+1
            edges=[e for e,count in counts.items() if count==1];indices=sorted({i for e in edges for i in e})
            matrix=inverse@np.array(obj.matrix_world);capture=verts@matrix[:3,:3].T+matrix[:3,3]
            xyz=capture[indices] if indices else np.empty((0,3));hom=xyz@K.T;uv=hom[:,:2]/hom[:,2,None] if len(xyz) else np.empty((0,2))
            inside=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<3248)&(uv[:,1]>=0)&(uv[:,1]<2464)
            rows.append({'template':case['template'],'sample_id':case['sample_id'],'object':obj.name,'parent':obj.parent.name if obj.parent else None,'vertices':len(verts),'boundary_edges':len(edges),'boundary_vertices':len(indices),'boundary_vertices_in_frame':int(inside.sum()),'local_bbox':[verts.min(0).tolist(),verts.max(0).tolist()],'capture_bbox':[capture.min(0).tolist(),capture.max(0).tolist()],'boundary_capture_points':xyz.tolist(),'boundary_native_px':uv.tolist(),'modifiers':[{'type':m.type,'name':m.name} for m in obj.modifiers]})
    (out/'REPORT.json').write_text(json.dumps({'scope':'mesh topology and projection; does not establish rendered visibility or final geometry repair','rows':rows},indent=2))
if __name__=='__main__':run()
