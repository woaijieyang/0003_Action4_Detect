"""New label revision from exact rays/faces; original candidate is retained."""
from pathlib import Path
import json,os
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_surface_intersection import exact_surface_hits
from model.training.action4_runtime import atomic_json

def run():
    source=R/'outputs/action4_surface_labels_v001';out=R/'outputs/action4_surface_labels_v002';out.mkdir(exist_ok=True)
    for split in ['train','eval','test']:contained_path(R,R/f'data/{split}/surface_v002_candidate',exists=False).mkdir(exist_ok=False)
    mesh=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz');vertices=mesh['vertices_mm']/1000;faces=mesh['faces'];rows=sum([json.loads((source/f'shard_{i}_COMPLETE.json').read_text())['rows'] for i in range(3)],[]);results=[]
    for row in rows:
        with np.load(contained_path(R,row['output'],exists=True)) as data:arrays={k:data[k] for k in data.files}
        tri=vertices[faces[arrays['face_indices']]];hit=exact_surface_hits(arrays['native_pixels'],arrays['K'],arrays['object_to_capture'],tri);valid=hit['valid'];T=arrays['object_to_capture'];xyz=hit['xyz_object_m'][valid].astype(np.float32);capture=xyz.astype(float)@T[:3,:3].T+T[:3,3];hom=capture@arrays['K'].T
        assert len(xyz)>0 and np.all(capture[:,2]>0)
        error=np.linalg.norm(hom[:,:2]/hom[:,2,None]-arrays['native_pixels'][valid],axis=1);assert error.max()<.005
        dropped=arrays['hit_grid_indices'][~valid];destination=R/f'data/{row["split"]}/surface_v002_candidate'/Path(row['output']).name
        with destination.open('xb') as f:np.savez_compressed(f,xyz_object_m=xyz,hit_grid_indices=arrays['hit_grid_indices'][valid],native_pixels=arrays['native_pixels'][valid],depth_capture_m=capture[:,2].astype(np.float32),face_indices=arrays['face_indices'][valid],missing_grid_indices=np.r_[arrays['missing_grid_indices'],dropped],bvh_missing_grid_indices=arrays['missing_grid_indices'],precision_rejected_grid_indices=dropped,K=arrays['K'],object_to_capture=T)
        results.append({**row,'source_candidate':row['output'],'output':str(destination.relative_to(R)),'hits':len(xyz),'misses':len(arrays['missing_grid_indices'])+len(dropped),'precision_rejected':len(dropped),'max_reprojection_error_px':float(error.max()),'bytes':destination.stat().st_size})
        if len(results)%100==0:atomic_json(out/'GENERATION_STATUS.json',{'completed':len(results),'total':len(rows)})
    atomic_json(out/'GENERATION_COMPLETE.json',{'status':'EXACT_INTERSECTION_CANDIDATE_PENDING_VALIDATION','rows':results,'parent_failure':'outputs/action4_surface_labels_v001/VALIDATION_FAILURE.json','source_candidate_retained':True,'training_accepted':False})
if __name__=='__main__':
    import traceback
    try:run()
    except BaseException:
        atomic_json(R/'outputs/action4_surface_labels_v002/GENERATION_FAILURE.json',{'traceback':traceback.format_exc()});raise
