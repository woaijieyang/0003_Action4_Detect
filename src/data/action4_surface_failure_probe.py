"""Locate surface/triangle mismatch without editing generated labels."""
from pathlib import Path
import json,numpy as np
from data.action4_paths import PROJECT_ROOT as R

def run():
    out=R/'outputs/action4_surface_labels_v001';mesh=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz');vertices=mesh['vertices_mm']/1000;faces=mesh['faces'];failures=[]
    rows=sum([json.loads((out/f'shard_{i}_COMPLETE.json').read_text())['rows'] for i in range(3)],[])
    for row in rows:
        data=np.load(R/row['output']);xyz=data['xyz_object_m'].astype(float);fi=data['face_indices'];tri=vertices[faces[fi]];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(normal,axis=1);error=np.abs(((xyz-tri[:,0])*normal).sum(1))/norm
        for i in np.where(error>=1e-6)[0]:
            tr32=tri[i].astype(np.float32).astype(float);n32=np.cross(tr32[1]-tr32[0],tr32[2]-tr32[0]);e32=float(abs(np.dot(xyz[i]-tr32[0],n32))/np.linalg.norm(n32))
            failures.append({'output':row['output'],'row':int(i),'grid_index':int(data['hit_grid_indices'][i]),'face_index':int(fi[i]),'plane_error_m':float(error[i]),'float32_mesh_plane_error_m':e32,'triangle_cross_norm':float(norm[i]),'triangle':tri[i].tolist(),'point':xyz[i].tolist(),'uv':data['native_pixels'][i].tolist()})
        if len(failures)>=5:break
    (out/'FAILURE_DETAILS.json').write_text(json.dumps({'scope':'diagnostic, labels unchanged','failures':failures},indent=2));print(json.dumps(failures[:5]))
if __name__=='__main__':run()
