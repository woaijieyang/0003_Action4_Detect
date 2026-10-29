import unittest,tempfile,json,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from model.training.action4_eval_publication import coordinated_publish
from model.training.action4_runtime import atomic_json
R=Path('/data/zy/vln/a-test');A='a'*32;C='b'*64
class PublicationTests(unittest.TestCase):
 def test_followers_wait_for_actual_publication(self):
  with tempfile.TemporaryDirectory(dir=R/'outputs') as d:
   out=Path(d);(out/'eval').mkdir();finished=threading.Event()
   def publish():time.sleep(.15);atomic_json(out/'eval/step_00000010.json',{'step':10,'config_id':C});finished.set()
   def call(rank):
    result=coordinated_publish(out,rank=rank,attempt=A,step=10,config_id=C,publish_fn=publish,timeout=2,poll=.005);self.assertTrue(finished.is_set());return result
   with ThreadPoolExecutor(4) as ex:results=list(ex.map(call,[1,2,3,0]))
   self.assertTrue(all(r['step']==10 for r in results))
 def test_failure_propagates_without_collective(self):
  with tempfile.TemporaryDirectory(dir=R/'outputs') as d:
   out=Path(d);(out/'eval').mkdir()
   def bad():raise ValueError('synthetic publication failure')
   with self.assertRaises(ValueError):coordinated_publish(out,rank=0,attempt=A,step=10,config_id=C,publish_fn=bad)
   with self.assertRaises(RuntimeError):coordinated_publish(out,rank=1,attempt=A,step=10,config_id=C,publish_fn=bad,timeout=.2,poll=.005)
 def test_release_identity_cannot_be_reused(self):
  with tempfile.TemporaryDirectory(dir=R/'outputs') as d:
   out=Path(d);(out/'eval').mkdir();(out/'publication').mkdir();atomic_json(out/'publication'/f'{A}_step_00000010.ready.json',{'attempt_id':A,'step':10,'config_id':'c'*64,'status':'PUBLISHED'})
   with self.assertRaises(ValueError):coordinated_publish(out,rank=1,attempt=A,step=10,config_id=C,publish_fn=lambda:None,timeout=.2,poll=.005)
