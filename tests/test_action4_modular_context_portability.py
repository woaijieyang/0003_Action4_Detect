"""Metadata fixed arithmetic and exact authorized-hunk regression; no Blender."""
from pathlib import Path
import ast,sys,unittest
from unittest.mock import patch
R=Path('/data/zy/vln/a-test');sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'src'))
from data import action4_modular_scene_sampling as m
class MetadataTests(unittest.TestCase):
 def test_scalar_rigid_inverse_product_no_NumPy_inverse_or_rounding(self):
  A=[[0.,-1.,0.,.125],[1.,0.,0.,-.25],[0.,0.,1.,.5],[0.,0.,0.,1.]]
  expected=[[0.,1.,0.,.25],[-1.,0.,0.,.125],[0.,0.,1.,-.5],[0.,0.,0.,1.]]
  with patch.object(m.np.linalg,'inv',side_effect=AssertionError('metadata must not use NumPy inverse')):
   inverse=m._metadata_rigid_inverse(A);self.assertEqual(inverse,expected)
   self.assertEqual(m._metadata_rigid_product(A,inverse),m.np.eye(4).tolist())
   self.assertTrue(all(type(v) is float for row in inverse for v in row))
  bad=[row[:] for row in A];bad[0][0]=2
  with self.assertRaises(ValueError):m._metadata_rigid_inverse(bad)
 def test_only_prepare_base_metadata_hunk_changes_original_function_AST(self):
  old=(R/'outputs/action4_modular_context_portability_v001/snapshot_v001/src/data/action4_modular_scene_sampling.py').read_text()
  current=Path(m.__file__).read_text()
  old_line="  source['sides'][side]={'A_brace_from_CAD':g['A'].tolist(),'S_skin_relative_to_brace':g['S'].tolist(),'F_body_from_brace':deepcopy(g['wc']['rigid_F_new_body']),'anchor_C_capture_by_frame':deepcopy(anchors),'base_C_capture':g['C'].tolist(),'base_B_capture':g['B'].tolist()}"
  new_lines="""  original=g['j'] if side=='left' else right['rc']
  metadata_B=_metadata_rigid_product(original['C_per_frame']['0'],_metadata_rigid_inverse(original['A_shared']))
  source['sides'][side]={'A_brace_from_CAD':g['A'].tolist(),'S_skin_relative_to_brace':g['S'].tolist(),'F_body_from_brace':deepcopy(g['wc']['rigid_F_new_body']),'anchor_C_capture_by_frame':deepcopy(anchors),'base_C_capture':g['C'].tolist(),'base_B_capture':metadata_B}"""
  self.assertEqual(old.count(old_line),1)
  expected=old.replace(old_line,new_lines)
  def nodes(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  a,b=nodes(expected),nodes(current)
  for name in a:self.assertEqual(a[name],b[name],name+' outside authorized metadata hunk')
  self.assertEqual(set(b)-set(a),{'_metadata_rigid_inverse','_metadata_rigid_product'})
if __name__=='__main__':unittest.main(verbosity=2)
