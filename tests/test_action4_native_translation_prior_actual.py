"""Actual native prior-only counterexample; not image tracking accuracy."""
import unittest
import numpy as np
import pym3t
from model.action4_current_pose_prior import native_prior_directions
class ActualTranslationPrior(unittest.TestCase):
 def solve(self, rotated):
  target=pym3t.Link("target",None,free_directions=[False]*6);assert target.SetUp()
  fixed=pym3t.Optimizer("fixed",target);assert fixed.SetUp()
  moving=pym3t.Link("moving",None);moving.maintain_rigid_pose=True;assert moving.SetUp()
  desired=np.eye(4);desired[0,3]=.01
  if rotated:desired[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
  target.link2world_pose=desired
  prior=pym3t.SoftConstraint("prior",target,moving,np.eye(4),np.eye(4),
    native_prior_directions("translation_only",True),0.,0.,.1,.01);assert prior.SetUp()
  opt=pym3t.Optimizer("opt",moving);assert opt.AddSoftConstraint(prior);assert opt.SetUp()
  assert opt.CalculateOptimization(0,0,0)
  return np.asarray(moving.link2world_pose)
 def test_rotation_of_anchor_does_not_pull_rotation(self):
  a=self.solve(False);b=self.solve(True)
  np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
  np.testing.assert_allclose(b[:3,:3],np.eye(3),atol=1e-8,rtol=0)
  self.assertGreater(b[0,3],0);self.assertLess(b[0,3],.01)
