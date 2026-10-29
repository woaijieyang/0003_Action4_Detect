import json,hashlib,tempfile,copy,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from data import action4_sam_native_export as e
from data import action4_sam_training_associations as a
from data.action4_sam_roi_inputs import array_digest
R=Path('/data/zy/vln/a-test')
class ValidationReuse(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=R/'outputs');self.root=Path(self.tmp.name)
  self.rgb=np.arange(8*9*3,dtype=np.uint8).reshape(8,9,3)
  self.p=self.root/'image.png';Image.fromarray(self.rgb).save(self.p)
  self.entry={'original_source':self.bind(self.p),'actual_input':self.bind(self.p),'native_shape':list(self.rgb.shape),'actual_pixels_sha256':array_digest(self.rgb)}
  e._VARIANT_PIXEL_PROOFS.clear();a._ANNOTATION_SUBSETS.clear()
 def tearDown(self):self.tmp.cleanup()
 def bind(self,p):return {'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 def test_pixel_reuse_and_mutated_expectation_rejected(self):
  with patch.object(e,'_v2_read',wraps=e._v2_read) as read:
   e._verify_variant_pixels(self.entry);n=read.call_count
   e._verify_variant_pixels(self.entry);self.assertEqual(n,read.call_count)
  bad=copy.deepcopy(self.entry);bad['native_shape']=[8,10,3]
  with self.assertRaises(ValueError):e._verify_variant_pixels(bad)
 def test_changed_pixel_file_rejected(self):
  e._verify_variant_pixels(self.entry)
  Image.fromarray(np.zeros_like(self.rgb)).save(self.p)
  with self.assertRaises(ValueError):e._verify_variant_pixels(self.entry)
 def test_annotation_subset_reuse_no_mutation_leak(self):
  p=self.root/'labels.jsonl';p.write_text(json.dumps({'sample_id':'a','values':[1]})+chr(10)+json.dumps({'sample_id':'b','values':[2]}))
  ref=self.bind(p)
  with patch.object(a,'_v2_read',wraps=a._v2_read) as read:
   rows,n=a._variant_annotations(ref,['a']);rows[0]['values'][0]=99
   again,total=a._variant_annotations(ref,['b','a'])
   self.assertEqual(read.call_count,1);self.assertEqual((n,total),(2,2));self.assertEqual(again[0]['values'],[1])
  with self.assertRaises(ValueError):a._variant_annotations(ref,['missing'])
  p.write_text('{}')
  with self.assertRaises(ValueError):a._variant_annotations(ref,['a'])
 def test_duplicate_outside_selected_domain_rejected(self):
  p=self.root/'dup.jsonl';p.write_text(chr(10).join(json.dumps({'sample_id':s}) for s in ['a','b','b']))
  with self.assertRaises(ValueError):a._variant_annotations(self.bind(p),['a'])
