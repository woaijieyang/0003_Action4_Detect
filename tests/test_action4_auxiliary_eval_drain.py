"""Child-process boundary tests, dummy files only."""
import unittest,tempfile,subprocess,sys
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R
class DrainTests(unittest.TestCase):
 def case(self,kind):
  root=R/'outputs/action4_eval_acceleration_v001';root.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=root) as d:
   d=Path(d);script=d/'probe.py';script.write_text("""import sys,os,signal,time
from pathlib import Path
sys.path.insert(0,'/data/zy/vln/a-test/src')
from eval.action4_auxiliary_eval_drain import install_drain
root=Path(sys.argv[1]);kind=sys.argv[2]
def commit(request):
 request.with_suffix('.done.json').write_text('{}')
 if kind=='error':raise RuntimeError('commit error after done')
def worker():
 request=None
 if kind=='idle':time.sleep(10);raise AssertionError('not drained')
 request=root/'request.json';commit(request);(root/'next_claim').write_text('bad')
install_drain(worker,commit,root/'DRAINED.json');os.kill(os.getpid(),signal.SIGUSR2)
try:worker()
except RuntimeError:(root/'ERROR_PRESERVED').write_text('yes');sys.exit(7)
""")
   p=subprocess.run([sys.executable,'-B',str(script),str(d),kind],capture_output=True,text=True)
   return p.returncode,(d/'DRAINED.json').exists(),(d/'next_claim').exists(),(d/'ERROR_PRESERVED').exists()
 def test_commit_boundary(self):self.assertEqual(self.case('commit'),(0,True,False,False))
 def test_idle_boundary(self):self.assertEqual(self.case('idle'),(0,True,False,False))
 def test_failure_not_hidden(self):self.assertEqual(self.case('error'),(7,False,False,True))
