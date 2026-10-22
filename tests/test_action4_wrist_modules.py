"""Small regressions for geometry extraction/frame and immutable source policy."""
import copy
import json
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_wrist_modules as m

class WristModulesTest(unittest.TestCase):
    def test_source_config(self):
        self.assertEqual(m.validate_config(json.loads((m.ROOT/'configs/data/action4_wrist_modules_v001.json').read_text()))['dataset_generation_allowed'],False)
    def test_no_generation(self):
        c=json.loads((m.ROOT/'configs/data/action4_wrist_modules_v001.json').read_text());c['dataset_generation_allowed']=True
        with self.assertRaises(ValueError):m.validate_config(c)
    def test_source_identity_required(self):
        c=json.loads((m.ROOT/'configs/data/action4_wrist_modules_v001.json').read_text());c['sources']['asset']['sha256']='0'*64
        with self.assertRaises(ValueError):m.validate_config(c)
    def test_near_rigid_roundoff(self):
        a=np.eye(4);a[0,0]+=1e-7;a[:3,3]=[1,2,3];q=m.rigid_frame(a)
        np.testing.assert_allclose(q[:3,:3].T@q[:3,:3],np.eye(3),rtol=0,atol=1e-14)
        p=np.array([[.3,.2,.1],[-1,2,0]])
        np.testing.assert_allclose(m.transform(q,m.transform(np.linalg.inv(q),p)),p,atol=1e-14)
    def test_mirror_rejected(self):
        a=np.eye(4);a[0,0]=-1
        with self.assertRaises(ValueError):m.rigid_frame(a)
    def test_scale_rejected(self):
        a=np.eye(4);a[0,0]=1.01
        with self.assertRaises(ValueError):m.rigid_frame(a)
    def test_invalid_matrix_rejected(self):
        for a in [np.zeros((3,3)),np.full((4,4),np.nan),np.diag([1,1,1,2])]:
            with self.assertRaises(ValueError):m.rigid_frame(a)
    def test_source_side_no_mirror(self):
        v=np.array([[1,0,0],[2,0,0],[1,1,0],[-1,0,0],[-2,0,0],[-1,1,0]])
        t=np.array([[0,1,2],[3,4,5],[0,3,2],[0,1,2]])
        r=np.array([102,93,102,21])
        self.assertEqual(m.selection(v,t,r,'left').tolist(),[0]);self.assertEqual(m.selection(v,t,r,'right').tolist(),[1])
        with self.assertRaises(ValueError):m.selection(v,t,r,'unknown')
    def test_single_boundary(self):
        f=np.array([[0,1,2],[0,2,3]])
        self.assertEqual(set(m.boundary_loop(f)),{0,1,2,3})
    def test_disconnected_openings_rejected(self):
        with self.assertRaises(ValueError):m.boundary_loop(np.array([[0,1,2],[3,4,5]]))
    def test_nonmanifold_rejected(self):
        with self.assertRaises(ValueError):m.boundary_loop(np.array([[0,1,2],[1,0,3],[0,1,4]]))

if __name__=='__main__':unittest.main()
