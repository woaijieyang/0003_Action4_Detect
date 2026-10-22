"""Analytic projection derivatives against central differences, not accuracy."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from model.action4_rigid_projection_analytic import projection_jacobian,project

def main():
    rng=np.random.default_rng(23);x=np.array([[a,b,c] for a in [-.035,.035] for b in [-.016,.016] for c in [-.022,.022]])
    radius=np.linalg.norm(x,axis=1).max();K=np.array([[900.,7.,224.],[0.,880.,220.],[0.,0.,1.]])
    largest=0
    for rot in [np.zeros(3),np.ones(3)*1e-8,np.array([3.14159,0,0])]+[rng.normal(size=3) for _ in range(12)]:
        p=np.r_[rot,.01,-.02,np.log(.5-radius)];weights=rng.uniform(.1,1,(8,1))
        def f(q):return ((project(x,Rotation.from_rotvec(q[:3]).as_matrix(),np.r_[q[3:5],radius+np.exp(q[5])],K))*weights).ravel()
        eps=1e-6;numeric=np.column_stack([(f(p+np.eye(6)[i]*eps)-f(p-np.eye(6)[i]*eps))/(2*eps) for i in range(6)])
        jac=projection_jacobian(p,x,radius,K,weights);error=float(np.max(np.abs(jac-numeric)));largest=max(largest,error);assert np.allclose(jac,numeric,atol=2e-5,rtol=2e-5),error
    out=Path('outputs/action4_geometry_speed_v001');out.mkdir(exist_ok=True);result={'status':'PASS','cases':15,'max_absolute_jacobian_error':largest,'scope':'derivative numerical check only'};(out/'JACOBIAN_TEST.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
