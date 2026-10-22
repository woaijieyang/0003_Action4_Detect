"""Performance changes must preserve pixels, effective batches and corruption rejection."""
import json,hashlib,tempfile,unittest
from pathlib import Path
import torch
from PIL import Image
from data.action4_dataset import Action4Dataset
from model.training.action4_trainer import EpochWindowSampler
from model.training.action4_dispatch import Dispatcher
ROOT=Path('/data/zy/vln/a-test')
class PerformanceTests(unittest.TestCase):
 def setUp(self):
  p=ROOT/'outputs/action4_training_performance_v001/test_cases';p.mkdir(exist_ok=True)
  self.root=Path(tempfile.mkdtemp(dir=p));self.rows=[]
  for i in range(4):
   name=f'{i}.png';Image.new('RGB',(64,64),(20+i,30+i,40+i)).save(self.root/name)
   self.rows.append({'schema_version':'action4.v1','sample_id':str(i),'image':name,'group_id':str(i),'background_source_id':'fixture','scene_family':'fixture','width':64,'height':64,'instances':[],'split':'train','status':'training_accepted','rgb_sha256':hashlib.sha256((self.root/name).read_bytes()).hexdigest()})
  self.manifest=self.root/'manifest.jsonl';self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
 def test_parallel_validation_identical_pixels_and_records(self):
  serial=Action4Dataset(self.root,self.manifest,split='train')
  parallel=Action4Dataset(self.root,self.manifest,split='train',validation_workers=4)
  self.assertEqual(serial.records,parallel.records)
  for i in range(4):self.assertTrue(torch.equal(serial[i]['image'],parallel[i]['image']))
 def test_parallel_validation_rejects_replaced_rgb(self):
  Image.new('RGB',(64,64),(1,2,3)).save(self.root/'1.png')
  with self.assertRaisesRegex(ValueError,'RGB hash mismatch'):Action4Dataset(self.root,self.manifest,validation_workers=4)
 def test_parallel_validation_still_completely_decodes_png(self):
  (self.root/'2.png').write_bytes((self.root/'2.png').read_bytes()[:40])
  with self.assertRaises((ValueError,OSError)):Action4Dataset(self.root,self.manifest,validation_workers=4)
 def test_cache_immutable_return_and_bounded_eviction(self):
  ds=Action4Dataset(self.root,self.manifest,validation_workers=2,decoded_cache_size=2)
  expected=ds[0]['image'].clone();ds[0]['image'].zero_()
  self.assertTrue(torch.equal(ds[0]['image'],expected))
  ds[1];ds[2];self.assertEqual(list(ds._decoded_cache),[1,2]);self.assertTrue(torch.equal(ds[0]['image'],expected))
 def test_cache_rejects_source_changes(self):
  ds=Action4Dataset(self.root,self.manifest,decoded_cache_size=2);ds[0]
  Image.new('RGB',(64,64),(1,2,3)).save(self.root/'0.png')
  with self.assertRaisesRegex(ValueError,'identity changed'):ds[0]
 def test_six_rank_exact_coverage_and_weighted_gradients(self):
  for global_batch,micro,n in [(32,6,35),(8,1,19),(8,2,17)]:
   ranks=[EpochWindowSampler(list(range(n)),global_batch,micro,r,6) for r in range(6)]
   batches=[list(x) for x in ranks];seen=[i for bs in batches for b in bs for i,real in b if real]
   self.assertEqual(sorted(seen),list(range(n)));self.assertEqual(len(set(len(x) for x in batches)),1)
   for wi,count in enumerate(ranks[0].window_counts):
    actual=0.
    for sampler,bs in zip(ranks,batches):
     w=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
     for b in bs[wi*sampler.accumulation:(wi+1)*sampler.accumulation]:
      x=torch.tensor([i+1 for i,real in b],dtype=torch.float64);real=sum(real for _,real in b)
      ((w*x).square().mean()*real*6/count).backward()
     actual+=w.grad.item()/6
    w=torch.tensor(.7,dtype=torch.float64,requires_grad=True);x=torch.arange(wi*global_batch+1,wi*global_batch+count+1,dtype=torch.float64);(w*x).square().mean().backward()
    self.assertAlmostEqual(actual,w.grad.item(),places=9)
 def test_six_rank_validation_and_loader_cpu_budget(self):
  cfg={'runtime':{'cpu_threads':2,'workers':10,'validation_workers':10,'stage_workers':{'D0':0}}}
  cp=self.root/'cfg.json';cp.write_text(json.dumps(cfg));d=Dispatcher(ROOT,self.root/'queue')
  _,_,minimum=d._effective_workload({'kind':'train'},ROOT/'scripts/action4_train.py',{'config':str(cp),'stage':'D0'},6)
  self.assertEqual(minimum,72)
if __name__=='__main__':unittest.main()
