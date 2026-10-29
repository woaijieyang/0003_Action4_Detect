import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path('/data/zy/vln/a-test')
spec=importlib.util.spec_from_file_location('delivery',ROOT/'scripts/action4_render_tracking_delivery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class VideoSourceTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs');self.root=Path(self.temp.name);self.old=m.R;m.R=self.root
 def tearDown(self):m.R=self.old;self.temp.cleanup()
 def write(self,p,v):
  p=self.root/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return p
 def setup_sources(self):
  replay=self.root/'candidate';baseline=self.root/'baseline'
  self.write('candidate/source/SNAPSHOT_MANIFEST.json',{'identity':'abc'})
  self.write('candidate/source/config.json',{'b0':{'sha256':'same'},'reanchor':{'baseline_job':'baseline'}})
  self.write('baseline/source/config.json',{'b0':{'sha256':'same'}})
  self.write('candidate/run/CLEANUP.json',{'failed':False})
  return replay,baseline
 def fn(self):
  self.assertTrue(callable(getattr(m,'resolve_bootstrap_source',None)),'generation-aware video source resolver missing')
  return m.resolve_bootstrap_source
 def complete(self):
  return {'status':'FULL2850_REANCHOR_CANDIDATE_COMPLETE_NOT_ACCURACY','frames':2850,'closed':True,'source_identity':'abc','events':[{'outcome':'native_bound'}]}
 def test_new_generation_video_keeps_correct_bootstrap_source(self):
  r,b=self.setup_sources()
  self.assertEqual(self.fn()(r,self.complete(),{}),b)
 def test_failed_cleanup_pending_events_or_changed_B0_refused(self):
  r,b=self.setup_sources()
  for key,v in [('closed',False),('source_identity','wrong'),('events',[{'outcome':'failed'}])]:
   c=self.complete();c[key]=v
   with self.assertRaises(ValueError):self.fn()(r,c,{})
  self.write('baseline/source/config.json',{'b0':{'sha256':'other'}})
  with self.assertRaises(ValueError):self.fn()(r,self.complete(),{})
 def test_legacy_replay_route_unchanged(self):
  r,b=self.setup_sources()
  self.assertEqual(self.fn()(r,{'frames':2850},{'native_job':'baseline'}),b)
if __name__=='__main__':unittest.main()
