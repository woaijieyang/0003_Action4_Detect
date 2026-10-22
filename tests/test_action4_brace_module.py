
import unittest,copy
from unittest.mock import patch
import numpy as np
from data import action4_brace_module as m
CONFIG='configs/data/action4_brace_module_v001.json'
class BraceModuleTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=m.load_config(CONFIG);cls.p,cls.V,cls.F,cls.M,cls.roles,cls.rims=m.expected(cls.c)
 def test_closed_oriented_annulus(self):
  F=self.F;e=np.r_[F[:,[0,1]],F[:,[1,2]],F[:,[2,3]],F[:,[3,0]]];u,inv,n=np.unique(np.sort(e,1),axis=0,return_inverse=True,return_counts=True)
  self.assertTrue(np.all(n==2));self.assertTrue(np.all(np.bincount(inv,weights=np.where(e[:,0]<e[:,1],1,-1))==0))
  self.assertEqual(len(self.V)-len(u)+len(F),0)
  tri=np.r_[F[:,[0,1,2]],F[:,[0,2,3]]];v=self.V
  volume=np.einsum('ij,ij->i',v[tri[:,0]],np.cross(v[tri[:,1]],v[tri[:,2]])).sum()/6
  self.assertGreater(volume,0)
 def test_normal_roles(self):
  n=np.cross(self.V[self.F[:,1]]-self.V[self.F[:,0]],self.V[self.F[:,2]]-self.V[self.F[:,0]]);c=self.V[self.F].mean(1)
  for normal,center,role in zip(n,c,self.roles):
   if role=='outer':self.assertGreater(normal@np.array([center[0],0,center[2]]),0)
   elif role=='inner':self.assertLess(normal@np.array([center[0],0,center[2]]),0)
   else:self.assertGreater(normal[1]*(1 if role=='distal_cap' else -1),0)
 def test_every_rim_sample_matches_frozen73points(self):
  ref=m.load(self.c['sources']['analytic34_reference']['path'])
  for name,ids in self.rims.items():
   actual=self.V[np.r_[ids,ids[0]]];self.assertEqual(len(actual),73)
   np.testing.assert_allclose(actual,ref['rings34_in_newreference'][name],rtol=0,atol=1e-14)
 def test_width_and_reference_not_green_overhang(self):
  self.assertAlmostEqual(np.ptp(self.V[:,1]),.034,places=14)
  for name,expected in [('proximal',-.049),('distal',-.015)]:
   np.testing.assert_allclose(self.V[self.rims[name],1]-.032,expected,rtol=0,atol=1e-14)
  self.assertTrue(np.all(abs(self.V[:,1])<=.017))
 def test_inner_outer_radii_and_radial_thickness(self):
  n=72
  np.testing.assert_allclose(self.V[:n]-self.V[4*n:5*n],np.c_[self.p['thickness']*np.cos(np.arange(n)*2*np.pi/n),np.zeros(n),self.p['thickness']*np.sin(np.arange(n)*2*np.pi/n)],atol=1e-14,rtol=0)
  self.assertTrue(np.all(self.M[:n]==1));self.assertTrue(np.all(self.M[n:2*n]==0));self.assertTrue(np.all(self.M[2*n:3*n]==1))
 def test_geometry_invalid_parameters_rejected(self):
  for args in [([0,.03],.0012,.034,72,.0024),([.03,.03],-.001,.034,72,.0024),([.03,.03],.0012,.034,72,.018),([np.nan,.03],.0012,.034,72,.0024)]:
   with self.assertRaises(ValueError):m.annular_geometry(*args)
 def test_unapproved_overrides_rejected(self):
  original=m.load
  for key,value in [('width_m',.028),('side','left'),('segments',48),('dataset_generation_allowed',True),('proximal_offset_from_wrist_m',.035)]:
   bad=copy.deepcopy(self.c);bad[key]=value
   with patch.object(m,'load',side_effect=lambda path:bad if path=='bad' else original(path)):
    with self.assertRaises(ValueError):m.load_config('bad')
if __name__=='__main__':unittest.main(verbosity=2)
