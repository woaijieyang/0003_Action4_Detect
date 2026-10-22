"""Read-only CAD-ray correspondence feasibility probe inside Blender."""
from pathlib import Path
import sys,json
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from data.action4_template_runtime import TemplateRuntime

def run():
    out=R/'outputs/action4_dense_surface_probe_v001';mesh=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz');vertices=mesh['vertices_mm']/1000;faces=mesh['faces']
    runtime=TemplateRuntime('wall');core=runtime.scene.objects['left_Action4_core'];actual=np.array([v.co[:] for v in core.data.vertices]);same=actual.shape==vertices.shape and np.allclose(actual,vertices,atol=1e-7,rtol=0)
    if not same:
        (out/'REPORT.json').write_text(json.dumps({'status':'MESH_BINDING_MISMATCH','actual_shape':actual.shape,'frozen_shape':vertices.shape}));return
    tree=BVHTree.FromPolygons([Vector(v) for v in vertices],faces.tolist(),all_triangles=True);rows=[]
    for obj in json.loads((out/'INPUTS.json').read_text())['rows']:
        K=np.array(obj['K']);T=np.array(obj['object_to_capture_cv_m']);inverse=np.linalg.inv(T);origin=inverse[:3,3];hits=[];misses=0;errors=[]
        for index,uv in zip(obj['grid_indices'],obj['native_pixels']):
            ray=inverse[:3,:3]@(np.linalg.inv(K)@np.r_[uv,1]);ray=ray/np.linalg.norm(ray);location,normal,face,distance=tree.ray_cast(Vector(origin),Vector(ray))
            if location is None:misses+=1;continue
            xyz=np.array(location);capture=T[:3,:3]@xyz+T[:3,3];h=K@capture;errors.append(float(np.linalg.norm(h[:2]/h[2]-uv)));hits.append({'grid_index':index,'xyz_object_m':xyz.tolist(),'depth_capture_m':float(capture[2]),'face_index':int(face)})
        rows.append({'sample_id':obj['sample_id'],'instance_id':obj['instance_id'],'mask_samples':obj['mask_samples'],'hits':hits,'miss_count':misses,'max_reprojection_error_px':max(errors) if errors else None})
    (out/'REPORT.json').write_text(json.dumps({'status':'DERIVED_CORRESPONDENCES_PROBE_ONLY','frozen_mesh_matches_render_core':True,'rows':rows,'training_labels_published':False,'caveat':'decal pixels and subpixel silhouette differences need validity masks; no invented coordinates for misses'},indent=2))
if __name__=='__main__':run()
