"""Synthetic morphology contracts and negative examples, not real semantic GT."""
import unittest
import numpy as np
from model.action4_sam_body_core import BodyCorePolicy,body_core_geometry

class BodyCoreTests(unittest.TestCase):
 def test_connected_thin_wire_removed_without_mutating_raw(self):
  m=np.zeros((220,360),bool);m[60:140,80:160]=True;m[98:102,160:310]=True;before=m.copy()
  r=body_core_geometry(m)
  np.testing.assert_array_equal(m,before);self.assertFalse((r['mask']&~m).any())
  self.assertTrue(r['receipt']['geometry_qualified']);self.assertLess(r['candidate_box'][2],180)
  self.assertGreaterEqual(r['receipt']['retained_fraction'],.5);self.assertFalse(r['receipt']['full_body_certified'])
 def test_equal_thick_components_rejected_even_with_prior(self):
  m=np.zeros((160,300),bool);m[40:120,20:100]=True;m[40:120,180:260]=True;m[78:82,100:180]=True
  r=body_core_geometry(m,prior_body_center=[60,80])
  self.assertTrue(r['receipt']['prior_used']);self.assertFalse(r['receipt']['geometry_qualified'])
  self.assertIn('ambiguous_thick_components',r['receipt']['ambiguity_reasons']);self.assertIsNone(r['qualified_box'])
 def test_prior_selects_component_but_cannot_certify_minor_mount(self):
  m=np.zeros((220,360),bool);m[30:170,20:160]=True;m[80:140,240:300]=True;m[108:112,160:240]=True
  r=body_core_geometry(m,prior_body_center=[270,110])
  self.assertTrue(r['receipt']['prior_used']);self.assertGreater(r['candidate_box'][0],200)
  self.assertFalse(r['receipt']['geometry_qualified']);self.assertIsNone(r['qualified_box'])
 def test_thick_connected_mount_is_explicit_negative_not_semantic_success(self):
  m=np.zeros((160,280),bool);m[40:120,40:120]=True;m[40:120,120:200]=True
  r=body_core_geometry(m)
  self.assertTrue(r['receipt']['geometry_qualified'])
  self.assertGreater(r['candidate_box'][2],180) # known synthetic body ends120; thick attachment survives.
  self.assertFalse(r['receipt']['full_body_certified']);self.assertEqual(r['receipt']['source'],'candidate_NOT_semantic_GT')
 def test_crop_translation_consistent(self):
  a=np.zeros((140,180),bool);a[30:100,30:100]=True;a[60:64,100:150]=True
  b=np.zeros((260,340),bool);b[70:210,80:260]=a
  x=body_core_geometry(a);y=body_core_geometry(b)
  np.testing.assert_array_equal(y['mask'][70:210,80:260],x['mask'])
  np.testing.assert_array_equal(np.array(y['candidate_box']),np.array(x['candidate_box'])+[80,70,80,70])
 def test_empty_thin_or_bad_inputs_fail_closed(self):
  r=body_core_geometry(np.zeros((20,30),bool));self.assertFalse(r['receipt']['geometry_qualified']);self.assertIsNone(r['qualified_box'])
  a=np.zeros((40,100),bool);a[20,10:90]=True;r=body_core_geometry(a);self.assertFalse(r['receipt']['geometry_qualified'])
  with self.assertRaises(ValueError):body_core_geometry(np.zeros((20,20),np.uint8))
  with self.assertRaises(ValueError):body_core_geometry(np.zeros((20,20),bool),prior_body_center=[float('nan'),2])
  with self.assertRaises(ValueError):BodyCorePolicy(minimum_retained_fraction=.49)
  with self.assertRaises(ValueError):BodyCorePolicy(radius_fraction=0)
 def test_elongated_thick_region_not_certified(self):
  a=np.zeros((100,500),bool);a[30:60,20:450]=True;r=body_core_geometry(a)
  self.assertFalse(r['receipt']['geometry_qualified']);self.assertIn('implausible_core_aspect',r['receipt']['ambiguity_reasons'])

if __name__=='__main__':unittest.main()
