"""Independent serialized correspondence verification against original manifests."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R,contained_path
from model.training.action4_runtime import atomic_json


def run(version="v001"):
    out=R/('outputs/action4_surface_labels_'+version)
    if (out/'GENERATION_COMPLETE.json').exists():rows=json.loads((out/'GENERATION_COMPLETE.json').read_text())['rows']
    else:
        exit_record=json.loads((out/'WORKERS_EXIT.json').read_text());assert exit_record['codes']==[0,0,0] and exit_record['all_receipts']
        rows=sum([json.loads((out/f'shard_{i}_COMPLETE.json').read_text())['rows'] for i in range(3)],[])
    lookup={(x['split'],x['sample_id'],x['instance_id']):x for x in rows};assert len(lookup)==len(rows)==3802
    mesh=np.load(R/'data/DJI_Action4_construct/r3_v001/mesh_v001.npz');vertices=mesh['vertices_mm']/1000;faces=mesh['faces'];details=[];expected=set()
    for split in ['train','eval','test']:
        for line in (R/f'data/{split}/annotations.jsonl').read_text().splitlines():
            record=json.loads(line)
            for obj in record['instances']:
                if obj['ignore']:continue
                key=(split,record['sample_id'],obj['instance_id']);expected.add(key);row=lookup[key];path=contained_path(R,row['output'],exists=True);assert path.parent==R/f'data/{split}/surface_{version}_candidate'
                with np.load(path) as data:
                    xyz=data['xyz_object_m'].astype(float);idx=data['hit_grid_indices'];missing=data['missing_grid_indices'];uv=data['native_pixels'];face=data['face_indices'];depth=data['depth_capture_m'];K=data['K'];T=data['object_to_capture']
                assert np.array_equal(K,np.array(record['camera']['K'])) and np.array_equal(T,np.array(obj['canonical8_geometry']['object_to_capture_cv_m']))
                union=np.r_[idx,missing];assert len(union)==len(set(union.tolist()))==row['mask_samples'] and np.all((union>=0)&(union<112*112));assert len(xyz)==len(idx)==len(face)==len(depth)>0
                box=np.array(obj['bbox_full_xyxy']);side=max(box[2:]-box[:2])*2.25;low=(box[:2]+box[2:])/2-side/2
                native=np.c_[union%112,union//112];native=low+(native+.5)*side/112;pixel=np.rint(native).astype(int)
                assert np.all((pixel[:,0]>=0)&(pixel[:,0]<record['width'])&(pixel[:,1]>=0)&(pixel[:,1]<record['height']))
                with Image.open(contained_path(R,obj['mask_visible'],exists=True)) as im:mask=np.asarray(im)
                assert np.all(mask[pixel[:,1],pixel[:,0]]>0)
                expected_uv=low+(np.c_[idx%112,idx//112]+.5)*side/112;assert np.allclose(uv,expected_uv,atol=1e-8,rtol=0)
                capture=xyz@T[:3,:3].T+T[:3,3];hom=capture@K.T;error=np.linalg.norm(hom[:,:2]/hom[:,2,None]-uv,axis=1)
                assert np.isfinite(xyz).all() and np.all(capture[:,2]>0) and np.allclose(depth,capture[:,2],atol=1e-6,rtol=0) and error.max()<=.005
                assert np.all((face>=0)&(face<len(faces)));tri=vertices[faces[face]];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];q=xyz-tri[:,0];normal=np.cross(a,b);norm=np.linalg.norm(normal,axis=1);assert np.all(norm>0)
                plane=np.abs((q*normal).sum(1))/norm;assert plane.max()<1e-6
                aa=(a*a).sum(1);ab=(a*b).sum(1);bb=(b*b).sum(1);qa=(q*a).sum(1);qb=(q*b).sum(1);den=aa*bb-ab*ab;assert np.all(den>0)
                u=(bb*qa-ab*qb)/den;v=(aa*qb-ab*qa)/den;assert np.min(np.c_[u,v,1-u-v])>-.005
                details.append({'split':split,'sample_id':record['sample_id'],'instance_id':obj['instance_id'],'hits':len(idx),'misses':len(missing),'max_reprojection_px':float(error.max()),'max_face_plane_error_m':float(plane.max()),'bytes':path.stat().st_size,'miss_fraction':len(missing)/len(union)})
                if len(details)%100==0:atomic_json(out/'VALIDATION_STATUS.json',{'verified_objects':len(details),'total':3802})
    assert expected==set(lookup)
    summary={'status':'SERIALIZED_GEOMETRY_PASS_VISUAL_REVIEW_PENDING','objects':len(details),'split_objects':{s:sum(x['split']==s for x in details) for s in ['train','eval','test']},'surface_points':sum(x['hits'] for x in details),'missing_points':sum(x['misses'] for x in details),'bytes':sum(x['bytes'] for x in details),'max_reprojection_px':max(x['max_reprojection_px'] for x in details),'max_face_plane_error_m':max(x['max_face_plane_error_m'] for x in details),'max_miss_fraction':max(x['miss_fraction'] for x in details),'details':details,'training_accepted':False,'source_RGB_modified':False}
    atomic_json(out/'VALIDATION.json',summary)
if __name__=='__main__':
    import traceback,argparse
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v001','v002'],default='v001');args=parser.parse_args()
    try:run(args.version)
    except BaseException:
        atomic_json(R/('outputs/action4_surface_labels_'+args.version)/'VALIDATION_FAILURE.json',{'traceback':traceback.format_exc()});raise
