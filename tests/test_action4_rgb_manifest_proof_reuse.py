import copy,unittest
from unittest.mock import patch
from tests.test_action4_rgb_variant_manifest import ManifestTests
from data import action4_rgb_variant_manifest as m
class ProofReuse(ManifestTests):
 def test_reuses_success_only_and_isolates_return_value(self):
  b=self.make();m._MANIFEST_PROOFS.clear()
  with patch.object(m,'_verify_manifest_uncached',wraps=m._verify_manifest_uncached) as full:
   a=m.verify_manifest(b);a['rows'].clear()
   result=m.verify_manifest(b)
   self.assertEqual(len(result['rows']),3);self.assertEqual(full.call_count,1)
 def test_failed_marker_invalidates_previous_proof(self):
  b=self.make();m.verify_manifest(b)
  (self.root/'pilot/FAILED.json').write_text('{}')
  with self.assertRaises(ValueError):m.verify_manifest(b)
 def test_new_process_requires_fresh_proof(self):
  b=self.make();m.verify_manifest(b)
  with patch.object(m.os,'getpid',return_value=-123),patch.object(m,'_verify_manifest_uncached',wraps=m._verify_manifest_uncached) as full:
   m.verify_manifest(b);self.assertEqual(full.call_count,1)
