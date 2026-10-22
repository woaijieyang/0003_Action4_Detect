"""Bind all template camera meshes to frozen CAD before surface-label generation."""
from pathlib import Path
import sys,json
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
import numpy as np
from data.action4_template_runtime import TemplateRuntime

def canonical_faces(faces):return sorted(map(tuple,np.sort(np.asarray(faces),axis=1).tolist()))
def run():
    out=R/'outputs/action4_surface_labels_v001';out.mkdir(exist_ok=True);mesh=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz');v=mesh['vertices_mm']/1000;f=canonical_faces(mesh['faces']);rows=[]
    for template in ['wall','table_full_garment','cloth_lift10_full_garment']:
        runtime=TemplateRuntime(template)
        for side in ['left','right']:
            obj=runtime.scene.objects[side+'_Action4_core'];actual=np.array([x.co[:] for x in obj.data.vertices]);faces=[list(x.vertices) for x in obj.data.polygons]
            same=actual.shape==v.shape and np.allclose(actual,v,atol=1e-7,rtol=0);triangles=all(len(x)==3 for x in faces);face_match=triangles and canonical_faces(faces)==f
            rows.append({'template':template,'side':side,'vertices_match':bool(same),'faces_match':bool(face_match),'modifiers':[m.type for m in obj.modifiers]})
    passed=all(x['vertices_match'] and x['faces_match'] and not x['modifiers'] for x in rows)
    (out/'MESH_BINDING.json').write_text(json.dumps({'status':'PASS' if passed else 'FAIL','scope':'all3 templates left/right core geometry; decals excluded from ray mesh, misses remain invalid','rows':rows},indent=2))
    if not passed:raise RuntimeError('Mesh binding failed')
if __name__=='__main__':run()
