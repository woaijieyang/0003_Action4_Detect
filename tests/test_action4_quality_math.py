"""NOT RUN. Small CPU/CUDA equivalence and direct counterexamples.
Run only after root accepts 128 and grants GPU3 quality execution.
"""
from copy import deepcopy
import importlib.util
import hashlib
import os
from pathlib import Path
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/zy/vln/a-test')
SELF_REF={'path':str((ROOT/'src/eval/action4_gpu_quality.py').relative_to(ROOT)), 'sha256':hashlib.sha256((ROOT/'src/eval/action4_gpu_quality.py').read_bytes()).hexdigest()}
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('gpu_quality_candidate',ROOT/'src/eval/action4_gpu_quality.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def geometry():
    xyz=[[-50,-30,-20],[50,-30,-20],[50,30,-20],[-50,30,-20],[-50,-30,20],[50,-30,20],[50,30,20],[-50,30,20]]
    T=np.eye(4);T[2,3]=-1.
    return {'points_object_mm':xyz,'object_to_world_m':T.tolist(),
            'camera_to_world_blender_m':np.eye(4).tolist(),
            'K_native':[[1700,0,1624],[0,1700,1232],[0,0,1]],'near_plane_m':.01}


class CPUKernelTests(unittest.TestCase):
    def test_cpu_torch_matches_original_project(self):
        original=m.load_original();g=geometry()
        want,depth=original.project8(g)
        got,actual=m.FP64Math('cpu').project8(g)
        np.testing.assert_allclose(got,want,rtol=0,atol=1e-10)
        np.testing.assert_allclose(actual,depth,rtol=0,atol=1e-12)

    def test_empty_pixel_set_and_outside_pixels(self):
        equations=np.array([[1,0,-10],[-1,0,0],[0,1,-10],[0,-1,0]],float)
        kernel=m.FP64Math('cpu',chunk_pixels=2)
        self.assertEqual(kernel.outside_hull(np.empty((0,2)),equations),0)
        xy=np.array([[0,0],[10,10],[5,5],[-1,5],[5,11]],float)
        self.assertEqual(kernel.outside_hull(xy,equations),2)
        self.assertEqual(kernel.hull_chunks,3)

    def test_near_plane_and_nonfinite_rejected(self):
        g=geometry();g['object_to_world_m'][2][3]=0
        with self.assertRaises(ValueError):m.FP64Math('cpu').project8(g)
        g=geometry();g['points_object_mm'][0][0]=float('nan')
        with self.assertRaises(ValueError):m.FP64Math('cpu').project8(g)

    def test_zero_presence_has_no_dummy_targets(self):
        original=m.load_original()
        entry={k:None for k in original.ENTRY_KEYS}
        entry.update(sample_id='synthetic_zero',target_count=0,target_presence=[False,False],split='train')
        row={**entry,'planned_target_count':0,'excluded_instances':[],'instances':[]}
        self.assertEqual(original.check_presence(row,entry),[])
        row['mask_visible']='dummy.png'
        with self.assertRaises(ValueError):original.check_presence(row,entry)


class CUDAKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.aid=m.GPUAid(implementation_ref=SELF_REF,chunk_pixels=7)  # Missing/wrong UUID or CUDA is a failure, not a skip.

    def test_project_matches_numpy_and_cpu_torch(self):
        original=m.load_original()
        for offset in (0.,.013,-.107):
            g=geometry();g['object_to_world_m'][0][3]=offset
            expected,depth=original.project8(g)
            got,z=self.aid.project8(g)
            cpu,z_cpu=m.FP64Math('cpu').project8(g)
            np.testing.assert_allclose(got,expected,rtol=0,atol=1e-10)
            np.testing.assert_allclose(got,cpu,rtol=0,atol=1e-10)
            np.testing.assert_allclose(z,depth,rtol=0,atol=1e-12)

    def test_hull_matches_original_numpy_including_threshold(self):
        from scipy.spatial import ConvexHull
        uv,_=m.load_original().project8(geometry())
        equations=ConvexHull(uv).equations
        rng=np.random.default_rng(37)
        xy=np.vstack((rng.uniform([1300,1000],[1950,1500],(101,2)),uv))
        expected=int(((xy@equations[:,:2].T+equations[:,2])>1e-6).any(axis=1).sum())
        self.assertEqual(self.aid.math.outside_hull(xy,equations),expected)
        eq=np.array([[1,0,-10],[-1,0,0],[0,1,-10],[0,-1,0]],float)
        xy=np.array([[10,5],[10+0.5e-6,5],[10+2e-6,5],[-1,5]],float)
        expected=int(((xy@eq[:,:2].T+eq[:,2])>1e-6).any(axis=1).sum())
        self.assertEqual(self.aid.math.outside_hull(xy,eq),expected)
        self.assertEqual(self.aid.math.outside_hull(np.empty((0,2)),eq),0)

    def test_gpu_work_receipt_is_actual(self):
        self.aid.project8(geometry())
        self.aid.math.outside_hull(np.array([[0.,0.],[11.,0.]]),np.array([[1.,0.,-10.]]))
        receipt=self.aid.receipt()
        self.assertGreater(receipt['projection_calls'],0)
        self.assertGreater(receipt['hull_pixels'],0)
        self.assertEqual(receipt['dtype'],'torch.float64')
        self.assertFalse(receipt['training_accepted'])


class MechanicalCopyTests(unittest.TestCase):
    def test_original_gates_preserved(self):
        import ast
        def extract(text,name):
            node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
            return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])
        expected=extract((ROOT/'src/eval/action4_corner_checks.py').read_text(),'validate_record')
        actual=extract((ROOT/'src/eval/action4_gpu_quality.py').read_text(),'validate_record_gpu')
        actual=actual.replace('def validate_record_gpu(reader,record,data,*,engine,original):\n bounded=original.bounded\n draw8=original.draw8\n project8=engine.project8','def validate_record(reader,record,data):')
        actual=actual.replace('   bad=engine.outside_hull(xy,equations)','   bad=0\n   for i in range(0,len(xy),50000):bad+=int(((xy[i:i+50000]@equations[:,:2].T+equations[:,2])>1e-6).any(axis=1).sum())')
        self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
