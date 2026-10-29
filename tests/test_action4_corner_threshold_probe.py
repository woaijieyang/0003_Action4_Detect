import copy,unittest,tempfile
from pathlib import Path
from eval.action4_corner_threshold_probe import variant_source,load_module,candidate_or_raw
from data.action4_paths import PROJECT_ROOT
class TestProbe(unittest.TestCase):
 def test_only_registered_q_and_schema_change(self):
  with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs/action4_corner_threshold_probe_v001') as tmp:
   original=(PROJECT_ROOT/'src/model/action4_single_frame_pose.py').read_text();base_path=Path(tmp)/'base.py';base_path.write_text(original);base=load_module(base_path,'test_base')
   for q in (.4,.2):
    path=Path(tmp)/('q'+str(q)+'.py');path.write_text(variant_source(original,q));candidate=load_module(path,'test_candidate')
    self.assertEqual({k for k in base.POLICY if base.POLICY[k]!=candidate.POLICY[k]},{'schema','q_corner_min'})
    with self.assertRaises(ValueError):candidate._policy(base.POLICY)
    self.assertEqual(candidate.POLICY['minimum_ids'],6)
   with self.assertRaises(ValueError):variant_source(original,.1)
 def test_fallback_preserves_raw_geometry_without_mutation(self):
  raw={'corners_px':[[1,2]],'rotation':[[1]],'translation_m':[2,3,4],'pose_valid':False,'published':True};old=copy.deepcopy(raw)
  result=candidate_or_raw(raw,{'reason':'insufficient_reliable_corner_ids','geometric_candidate_accepted':False})
  self.assertEqual(raw,old)
  self.assertTrue(all(result[k]==v for k,v in old.items()))
  self.assertIs(result['pose_diagnostic']['deployed'],False)
if __name__=='__main__':unittest.main()
