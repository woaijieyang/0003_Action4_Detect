import importlib.util,sys,unittest,dataclasses,json
from pathlib import Path
import numpy as np
import model.action4_local_evidence as new
from model.action4_guidance_contract import FrameIdentity,SourceIdentity,CandidateRecord
ROOT=Path('/data/zy/vln/a-test')
sp=importlib.util.spec_from_file_location('model._local_before_performance',ROOT/'outputs/action4_local_evidence_performance_v001/local_evidence_before.py');old=importlib.util.module_from_spec(sp);sys.modules[sp.name]=old;sp.loader.exec_module(old)
class EquivalenceTests(unittest.TestCase):
 def test_integer_fractional_negative_and_outside_boundaries(self):
  rng=np.random.default_rng(17);mask=rng.random((23,31))>.7
  boxes=[(-3.4,-5.2,5.7,10.1),(4.,4.,12.,12.),(40,40,50,50),(-10,-8,-2,-1),(0,0,31,23)]
  for _ in range(300):
   xy=rng.uniform(-40,50,2);wh=rng.uniform(.01,50,2);boxes.append(tuple(np.r_[xy,xy+wh]))
  for b in boxes:self.assertEqual(old._inside(mask,b),new._inside(mask,b))
 def test_extreme_finite_off_image_bounds_do_not_overflow(self):
  mask=np.ones((3,4),bool)
  for b in [(-1e30,-1e30,1e30,1e30),(1e30,1e30,2e30,2e30),(-2e30,-2e30,-1e30,-1e30)]:
   self.assertEqual(old._inside(mask,b),new._inside(mask,b))
 def test_complete_diagnostics_and_evidence_equal(self):
  rng=np.random.default_rng(18);frame=FrameIdentity('f',0,'native',31,23,'a'*64);bs=SourceIdentity('b0','b'*64,'c'*64);ys=SourceIdentity('yolo_visual','d'*64,'e'*64,'p','f'*64)
  for case in range(20):
   masks=[rng.random((23,31))>threshold for threshold in [.2,.5,.8]];masks.append(masks[0].copy());masks.append(np.zeros((23,31),bool))
   sources=tuple(CandidateRecord(frame,str(i),str(i),ys,(1.2,2.4,29.8,21.3),(('score',.8),),support_kind='mask',source_mask_ref='m'+str(i),evidence_channel='negative' if i==2 else 'visual_positive',evidence_score=.8) for i in range(5))
   boxes=[(-3.4,-2,15.5,17.5),(2,3,16,18),(12,8,30,22)]
   targets=tuple(CandidateRecord(frame,'b'+str(i),'b'+str(i),bs,b,(('score',.7),)) for i,b in enumerate(boxes))+sources[:2]
   def run(module):
    supports=tuple(module.PredictedSupport(frame,'m'+str(i),m) for i,m in enumerate(masks))
    return dataclasses.asdict(module.build_local_evidence(targets,sources,supports))
   self.assertEqual(run(old),run(new))
if __name__=='__main__':unittest.main(verbosity=2)
