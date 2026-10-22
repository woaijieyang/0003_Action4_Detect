"""Measured custom-normal roundtrip regression; CPU arrays only, no Blender."""
from pathlib import Path
import sys,unittest,ast,hashlib,numpy as np
sys.dont_write_bytecode=True
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_modular_composed_render as module
D=np.load(R/'outputs/action4_composed_normal_diagnosis_v001/normal_arrays.npz',allow_pickle=False)
class NormalRoundtripTests(unittest.TestCase):
 def accepts(self,a,b):
  fn=getattr(module,'_verify_source_normal_roundtrip',None)
  if fn is None:return False
  try:fn(a,b);return True
  except ValueError:return False
 def test_measured_Y25_stored_normals_accepted(self):
  self.assertTrue(self.accepts(D['actual_normals'],D['target_Y25_normals']),'Measured Y25 must satisfy inherited angular storage contract')
 def test_existing_frozen_neutral_stored_normals_accepted(self):
  self.assertTrue(self.accepts(D['neutral_library_actual_normals'],D['neutral_source_target_normals']),'Original frozen neutral library must remain accepted')
 def test_real_direction_and_mapping_errors_rejected(self):
  A,B=D['actual_normals'],D['target_Y25_normals'];one=B.copy();one[0]*=-1
  i=int(np.argmax(D['normalized_angles_rad']));rotated=B.copy();axis=np.cross(B[i],[1,0,0]);axis/=np.linalg.norm(axis);theta=.01;rotated[i]=B[i]*np.cos(theta)+np.cross(axis,B[i])*np.sin(theta)+axis*np.dot(axis,B[i])*(1-np.cos(theta))
  for bad in [np.roll(B,1,axis=0),np.roll(B.reshape(-1,3,3),1,axis=1).reshape(-1,3),-B,D['neutral_source_target_normals'],one,rotated]:
   with self.subTest(max_component_difference=float(abs(A-bad).max())):self.assertFalse(self.accepts(A,bad))
 def test_invalid_shapes_nonfinite_nonunit_rejected(self):
  A,B=D['actual_normals'],D['target_Y25_normals'];nan=A.copy();nan[0,0]=np.nan
  for bad in [A[:-1],nan,A*2,np.zeros_like(A)]:
   with self.subTest(shape=bad.shape):self.assertFalse(self.accepts(bad,B))
if __name__=='__main__':unittest.main(verbosity=2)
