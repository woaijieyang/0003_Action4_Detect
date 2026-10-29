import unittest,json,hashlib
from pathlib import Path
import numpy as np
from eval.action4_reference_probe_metrics import metrics,auc,normalize,bank_candidates,image_domain_counts
R=Path('/data/zy/vln/a-test');OUT=R/'outputs/action4_reference_evidence_candidate_v001/test_fixtures'
class ReferenceMetricsTests(unittest.TestCase):
 def test_metrics_and_auc_denominators(self):
  self.assertEqual(metrics([.9,.8,.1],[1,0,1],.5),{'count':3,'positive':2,'negative':1,'tp':1,'fp':1,'fn':1,'precision':.5,'recall':.5,'F1':.5})
  self.assertEqual(auc([.2,.8],[0,1]),1);self.assertEqual(auc([.8,.2],[0,1]),0);self.assertEqual(auc([.5,.5],[0,1]),.5)
  self.assertIsNone(auc([.5],[1]));self.assertIsNone(metrics([],[],.5)['F1'])
  with self.assertRaises(ValueError):metrics([.5],[-1],.5)
 def test_zero_feature_stays_zero_and_wrong_shape_rejected(self):
  f=np.zeros((2,128),np.float32);f[1,0]=2;z=normalize(f);self.assertTrue(np.array_equal(z[0],f[0]));self.assertEqual(z[1,0],1)
  with self.assertRaises(ValueError):normalize(np.ones((3,144)))
 def test_bank_uses_only_train_and_ignores_mixed(self):
  OUT.mkdir(exist_ok=True);p=OUT/'bank_metric_fixture.npz';f=np.ones((5,128),np.float32)
  np.savez(p,appearance=f,labels=np.array([1,1,0,0,-1]),target=np.array([0,0,-1,-1,-1]),center_score=np.array([.3,.9,.2,.8,1.]),grid=np.arange(5))
  row={'split':'train','file':str(p),'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'image_id':'trainonly','scene':'s'}
  bank=bank_candidates({'records':[row,dict(row,split='eval',file='not_a_real_file')]})
  self.assertEqual([(x['grid'],x['label']) for x in bank],[(1,1),(3,0)])
 def test_full_image_domain_cannot_hide_ignored_candidates(self):
  row={'image_id':'i','scene':'s','GT_count':2,'candidates':5,'positive':1,'negative':1,'ignored_or_ambiguous':3}
  d=image_domain_counts(row,np.array([1,0,-1,-1,-1]));self.assertEqual(d['eligible'],2);self.assertEqual(d['all_candidates'],5);self.assertEqual(d['GT_count'],2)
  with self.assertRaises(ValueError):image_domain_counts(dict(row,ignored_or_ambiguous=0),np.array([1,0,-1,-1,-1]))

if __name__=='__main__':unittest.main(verbosity=2)
