import sys,tempfile,unittest
from pathlib import Path
R=Path('/data/zy/vln/a-test');sys.path[:0]=[str(R/'src'),str(Path(__file__).parent)]
from PIL import Image
from data import action4_adapter as m
from data.action4_png_admission import ParallelPngAdmission
class Tests(unittest.TestCase):
 def setUp(self):
  base=R/'outputs/action4_png_admission_tests';base.mkdir(exist_ok=True);self.d=Path(tempfile.mkdtemp(dir=base));self.p=self.d/'a.png';Image.new('RGB',(8,8),(1,2,3)).save(self.p);self.size=m._validated_png_size;self.sha=m._sha256
 def test_exact_size_hash_and_fresh_closing(self):
  original=self.sha(self.p)
  with ParallelPngAdmission(R,[self.p],2) as c:
   self.assertEqual(m._validated_png_size(self.p),(8,8));self.assertEqual(m._sha256(self.p),original)
   calls=[];old=c.old_sha;c.old_sha=lambda p:(calls.append(p),old(p))[1]
   self.assertEqual(m._sha256(self.p),original);self.assertEqual(calls,[self.p]);self.assertEqual(c.calls[self.p],2)
  self.assertIs(m._sha256,self.sha);self.assertIs(m._validated_png_size,self.size)
 def test_mutation_rejected_and_helpers_restored(self):
  with self.assertRaisesRegex(ValueError,'changed'):
   with ParallelPngAdmission(R,[self.p],2):
    Image.new('RGB',(8,8),(4,5,6)).save(self.p);m._sha256(self.p)
  self.assertIs(m._sha256,self.sha)
 def test_corrupt_png_rejected_before_patch(self):
  self.p.write_bytes(b'bad')
  with self.assertRaises((ValueError,OSError)):
   with ParallelPngAdmission(R,[self.p],2):pass
  self.assertIs(m._sha256,self.sha)
 def test_duplicate_paths_rejected(self):
  with self.assertRaises(ValueError):ParallelPngAdmission(R,[self.p,self.p])
 def test_non_input_sha_not_cached(self):
  p=self.d/'other.txt';p.write_text('a')
  with ParallelPngAdmission(R,[self.p],2):
   a=m._sha256(p);p.write_text('b');self.assertNotEqual(a,m._sha256(p))
if __name__=='__main__':unittest.main()
