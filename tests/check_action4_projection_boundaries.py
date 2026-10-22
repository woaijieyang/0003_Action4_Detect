"""Reference/analytic solver gates and fixed cuboid boundary comparisons."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from model.action4_rigid_projection import fit_rigid_cuboid as old,project
from model.action4_rigid_projection_analytic import fit_rigid_cuboid as new

def main():
    rng=np.random.default_rng(29);xyz=np.array([[a,b,c] for a in [-.035,.035] for b in [-.016,.016] for c in [-.022,.022]])
    K=np.array([[600.,0,224.],[0,610.,224.],[0,0,1.]])
    uv=project(xyz,np.eye(3),[0,0,.6],K)
    cases=[('no_confidence',uv,np.zeros(8)),('three_confident',uv,np.r_[np.ones(3),np.zeros(5)]),('collapsed',np.ones((8,2))*100,np.ones(8)),('collinear',np.c_[np.arange(8)*10,np.ones(8)*100],np.ones(8))]
    for i,angle in enumerate([0.,1e-8,3.14159]):
        pts=project(xyz,Rotation.from_rotvec([angle,.2,-.1]).as_matrix(),[.03,-.01,.4+i*.4],K)+rng.normal(0,.15,(8,2));cases.append((f'pose_{i}',pts,np.linspace(.2,1,8)))
    rows=[]
    for name,pts,w in cases:
        a,b=old(pts,xyz,K,w),new(pts,xyz,K,w);assert a['accepted']==b['accepted'] and a['reason']==b['reason'],name
        delta=None
        if a['fitted_corners_px'] is not None:
            delta=float(np.max(np.abs(np.array(a['fitted_corners_px'])-b['fitted_corners_px'])));assert delta<.001,(name,delta)
        rows.append({'case':name,'reason':a['reason'],'max_difference_px':delta})
    out=Path('outputs/action4_geometry_speed_v001');(out/'BOUNDARY_TEST.json').write_text(json.dumps({'status':'PASS','cases':rows},indent=2));print(json.dumps(rows))
if __name__=='__main__':main()
