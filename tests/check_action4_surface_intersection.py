"""Regression for measured BVH hit-plane discrepancy; same bound before/after."""
from pathlib import Path
import json,argparse
import numpy as np

def main(corrected):
    root=Path.cwd();case=json.loads((root/'outputs/action4_surface_labels_v001/FAILURE_DETAILS.json').read_text())['failures'][0];data=np.load(root/case['output']);i=case['row'];tri=np.array(case['triangle']);point=data['xyz_object_m'][i].astype(float)
    if corrected:
        from data.action4_surface_intersection import exact_surface_hits
        result=exact_surface_hits(data['native_pixels'][i:i+1],data['K'],data['object_to_capture'],tri[None]);assert result['valid'][0];point=result['xyz_object_m'][0].astype(np.float32).astype(float)
    normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);plane=abs(np.dot(point-tri[0],normal))/np.linalg.norm(normal)
    assert plane<1e-6,f'point off reported face: {plane}m'
    capture=data['object_to_capture'][:3,:3]@point+data['object_to_capture'][:3,3];h=data['K']@capture;error=np.linalg.norm(h[:2]/h[2]-data['native_pixels'][i]);assert error<.005
    print(json.dumps({'status':'PASS','corrected':corrected,'plane_error_m':plane,'reprojection_px':error}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corrected',action='store_true');a=p.parse_args();main(a.corrected)
