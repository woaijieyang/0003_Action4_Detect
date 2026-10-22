"""Generate CAD-surface supervision sidecars; no measured-depth claim."""
from pathlib import Path
import sys,json,os
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from data.action4_template_runtime import bounded

def run(shard):
    out=R/'outputs/action4_surface_labels_v001';mesh=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz');tree=BVHTree.FromPolygons([Vector(v) for v in mesh['vertices_mm']/1000],mesh['faces'].tolist(),all_triangles=True)
    specs=json.loads((out/'INPUTS_COMPLETE.json').read_text())['rows'][shard::3];summaries=[]
    for row in specs:
        source=np.load(bounded(row['input']));T=source['object_to_capture'];K=source['K'];inverse=np.linalg.inv(T);origin=inverse[:3,3];rays=np.c_[source['native_pixels'],np.ones(len(source['grid_indices']))]@np.linalg.inv(K).T@inverse[:3,:3].T;rays/=np.linalg.norm(rays,axis=1)[:,None]
        hits=[];indices=[];faces=[];depths=[];errors=[];uvs=[];miss=[]
        for grid,uv,ray in zip(source['grid_indices'],source['native_pixels'],rays):
            location,normal,face,distance=tree.ray_cast(Vector(origin),Vector(ray))
            if location is None:miss.append(int(grid));continue
            xyz=np.array(location);capture=T[:3,:3]@xyz+T[:3,3];h=K@capture;error=float(np.linalg.norm(h[:2]/h[2]-uv))
            if not np.isfinite(xyz).all() or capture[2]<=0 or error>.005:raise ValueError('Invalid surface projection '+row['sample_id'])
            hits.append(xyz);indices.append(grid);faces.append(face);depths.append(capture[2]);errors.append(error);uvs.append(uv)
        path=bounded(row['output']);assert not path.exists();temporary=path.with_suffix('.tmp')
        with temporary.open('xb') as f:np.savez_compressed(f,xyz_object_m=np.array(hits,dtype=np.float32).reshape(-1,3),hit_grid_indices=np.array(indices,dtype=np.int32),native_pixels=np.array(uvs,dtype=np.float64).reshape(-1,2),depth_capture_m=np.array(depths,dtype=np.float32),face_indices=np.array(faces,dtype=np.int32),missing_grid_indices=np.array(miss,dtype=np.int32),K=K,object_to_capture=T)
        os.replace(temporary,path);summaries.append({**row,'hits':len(hits),'misses':len(miss),'max_reprojection_error_px':max(errors) if errors else None,'bytes':path.stat().st_size})
        if len(summaries)%50==0:(out/f'shard_{shard}_STATUS.json').write_text(json.dumps({'completed':len(summaries),'total':len(specs)}))
    (out/f'shard_{shard}_COMPLETE.json').write_text(json.dumps({'status':'GENERATED_CANDIDATE_NOT_TRAINING_ACCEPTED','rows':summaries},indent=2))
if __name__=='__main__':run(int(sys.argv[sys.argv.index('--')+1]))
