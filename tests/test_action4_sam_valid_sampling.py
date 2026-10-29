"""Exact nearest-kernel equivalence; no approximation of half-pixel ties."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
import data.action4_sam_roi_inputs as roi
from tests import test_action4_sam_roi_inputs as fixtures

def original(mask,A):
    a=np.asarray(mask);matrix=np.asarray(A,float)
    yy,xx=torch.meshgrid(torch.arange(448,dtype=torch.float32),torch.arange(448,dtype=torch.float32),indexing="ij")
    destination=torch.stack((xx,yy,torch.ones_like(xx)),dim=-1)
    xy=destination@torch.as_tensor(np.linalg.inv(matrix),dtype=torch.float32).T
    grid=torch.stack((2*(xy[...,0]+.5)/a.shape[1]-1,2*(xy[...,1]+.5)/a.shape[0]-1),dim=-1)
    return F.grid_sample(torch.as_tensor(a.astype(np.float32))[None,None],grid[None],
        mode="nearest",padding_mode="zeros",align_corners=False)[0]

class ValidSamplingTests(unittest.TestCase):
    def test_zero_stride_equals_original_for_padding_ties_and_affines(self):
        matrices=[np.eye(3)]
        for x in (-448.,-1.5,-.5,0.,.5,1.5,448.,np.nextafter(.5,1.)):
            matrices.append(np.array([[1.,0.,x],[0.,1.,-.5],[0.,0.,1.]]))
        matrices.extend([np.array([[.47,.013,-20.25],[-.007,.47,13.75],[0.,0.,1.]]),
                         np.array([[23.,0.,15.5],[0.,23.,-20.5],[0.,0.,1.]])])
        for shape in ((1,1),(7,11),(448,448),(2464,3248)):
            for A in matrices:
                with self.subTest(shape=shape,A=A.tolist()):
                    grid=roi._native_nearest_grid(shape,A)
                    actual=roi._sample_native_valid_on_grid(shape,grid)
                    self.assertTrue(torch.equal(actual,original(np.ones(shape,np.uint8),A)))
    def test_native_mask_original_exact_and_input_unchanged(self):
        rng=np.random.default_rng(19);a=rng.random((71,93))>.5;before=a.copy()
        A=np.array([[2.3,.03,-50.5],[.01,2.3,9.5],[0.,0.,1.]])
        self.assertTrue(torch.equal(roi.sample_native_nearest(a,A),original(a,A)))
        np.testing.assert_array_equal(a,before)
    def test_external_nonbinary_nonfinite_and_bad_matrix_rejected(self):
        for a in (np.full((3,4),np.nan),np.full((3,4),np.inf),np.full((3,4),2),np.full((3,4),.5),np.empty((0,4))):
            with self.assertRaises(ValueError):roi.sample_native_nearest(a,np.eye(3))
        for A in (np.zeros((3,3)),np.full((3,3),np.nan),np.eye(2)):
            with self.assertRaises(ValueError):roi.sample_native_nearest(np.ones((3,4),bool),A)
    def test_valid_source_is_four_bytes_zero_stride(self):
        real=F.grid_sample;seen=[]
        def capture(source,grid,**kwargs):
            seen.append((source.untyped_storage().nbytes(),source.stride(),source.shape))
            return real(source,grid,**kwargs)
        with patch.object(F,"grid_sample",side_effect=capture):
            roi._sample_native_valid_on_grid((2464,3248),roi._native_nearest_grid((2464,3248),np.eye(3)))
        self.assertEqual(seen[0][0],4);self.assertEqual(seen[0][1][-2:],(0,0))
    def test_full_inputs_equal_dense_valid_reference_and_grid_built_once(self):
        f=fixtures.SamRoiInputTests();f.setUp();prediction=f.pred()
        def dense(shape,grid):
            return F.grid_sample(torch.ones(1,1,*shape),grid[None],mode="nearest",padding_mode="zeros",align_corners=False)[0]
        with patch.object(roi,"_sample_native_valid_on_grid",side_effect=dense):reference=f.build(prediction)
        with patch.object(roi,"_native_nearest_grid",wraps=roi._native_nearest_grid) as grid:
            actual=f.build(prediction);self.assertEqual(grid.call_count,1)
        for name in ("rgb","valid_image_support"):self.assertTrue(torch.equal(getattr(actual,name),getattr(reference,name)))
        for name in ("visible","distance","available","candidate_overlap","dropout"):
            self.assertTrue(torch.equal(getattr(actual.mask_aux,name),getattr(reference.mask_aux,name)))
        for name in ("A448","A112","A224"):np.testing.assert_array_equal(getattr(actual,name),getattr(reference,name))
        for name in ("K112","K224"):
            for a,b in zip(getattr(actual,name),getattr(reference,name)):np.testing.assert_array_equal(a,b)
        self.assertEqual(actual.source_dependencies,reference.source_dependencies)
        self.assertEqual(actual.reasons,reference.reasons)

if __name__=="__main__":unittest.main()
