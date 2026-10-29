import unittest,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from data.action4_reference_probe import labels_after_prediction,coverage,selected_ids
R=Path('/data/zy/vln/a-test');OUT=R/'outputs/action4_reference_evidence_candidate_v001/test_fixtures'
class ReferenceLabelsTests(unittest.TestCase):
 def fixture(self,box,mask,name,ignored=False):
  OUT.mkdir(exist_ok=True);p=OUT/(name+'.png');Image.fromarray(mask.astype(np.uint8)*255).save(p)
  return {'bbox_full_xyxy':box,'mask_visible':str(p),'mask_visible_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ignore':ignored}
 def test_known_positive_background_and_ambiguous(self):
  m=np.zeros((20,20),bool);m[2:8,2:8]=True;g=self.fixture([2,2,8,8],m,'body')
  boxes=np.array([[2,2,8,8],[12,12,18,18],[0,0,20,20]],float)
  label,target,_,cov=labels_after_prediction(boxes,{'instances':[g],'height':20,'width':20})
  self.assertEqual(label.tolist(),[1,0,-1]);self.assertEqual(target.tolist(),[0,-1,-1]);self.assertEqual(cov[-1],1)
 def test_two_targets_ignore_and_invisible_not_false_labels(self):
  m=np.zeros((20,20),bool);m[2:8,2:8]=True;a=self.fixture([2,2,8,8],m,'a');b=self.fixture([2,2,8,8],m,'b')
  label,*_=labels_after_prediction(np.array([[2,2,8,8]],float),{'instances':[a,b],'height':20,'width':20});self.assertEqual(label[0],-1)
  invisible=self.fixture([2,2,8,8],np.zeros_like(m),'hidden')
  label,*_=labels_after_prediction(np.array([[2,2,8,8]],float),{'instances':[invisible],'height':20,'width':20});self.assertEqual(label[0],-1)
  ignored=dict(a,ignore=True)
  label,*_=labels_after_prediction(np.array([[2,2,8,8]],float),{'instances':[ignored],'height':20,'width':20});self.assertEqual(label[0],-1)
 def test_integral_coverage_matches_integer_center_selection(self):
  rng=np.random.default_rng(17);m=rng.random((23,31))>.7;xy=rng.uniform(-40,40,(200,2));wh=rng.uniform(.01,50,(200,2));boxes=np.concatenate([xy,xy+wh],1);actual,total=coverage(boxes,m);yy,xx=np.nonzero(m)
  expected=[np.count_nonzero((xx>=l)&(xx<r)&(yy>=t)&(yy<b))/len(xx) for l,t,r,b in boxes];np.testing.assert_allclose(actual,expected,atol=0,rtol=0)
 def test_mixed_with_second_body_below_IoU50_is_not_positive(self):
  a=np.zeros((20,20),bool);a[:10,:8]=True;b=np.zeros_like(a);b[:10,8:12]=True
  ga=self.fixture([0,0,10,10],a,'mixedA');gb=self.fixture([8,0,12,10],b,'mixedB')
  for other in [gb,self.fixture([8,0,12,10],np.zeros_like(a),'mixedHidden')]:
   label,*_=labels_after_prediction(np.array([[0,0,10,10]],float),{'instances':[ga,other],'height':20,'width':20});self.assertEqual(label[0],-1)
 def test_selection_unique_and_exact_seed_scene_contract(self):
  records=[{'sample_id':str(i),'scene_family':str(i%3)} for i in range(5200)];rng=np.random.default_rng(123);expected=[]
  for group in ['0','1','2']:expected+=sorted(rng.choice([r['sample_id'] for r in records if r['scene_family']==group],64,replace=False).tolist())
  cfg={'seed':123,'selection_protocol':'seeded_sorted_scene64_v1','train_image_ids':expected}
  self.assertEqual(selected_ids(records,'train',cfg),expected)
  with self.assertRaises(ValueError):selected_ids(records,'train',dict(cfg,train_image_ids=[expected[0]]*192))
  with self.assertRaises(ValueError):selected_ids(records,'train',dict(cfg,train_image_ids=expected[::-1]))
  with self.assertRaises(ValueError):selected_ids(records[:-1]+[records[0]],'train',cfg)
  ev=records[:650];self.assertEqual(selected_ids(ev,'eval',cfg),[r['sample_id'] for r in ev])
  with self.assertRaises(ValueError):selected_ids(ev[:-1]+[ev[0]],'eval',cfg)

if __name__=='__main__':unittest.main(verbosity=2)
