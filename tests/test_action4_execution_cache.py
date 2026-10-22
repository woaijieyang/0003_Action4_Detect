"""Execution-cache regression: bounded fixtures, no GPU or formal acceptance."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from data import action4_execution_cache as cache

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=cache.R/'outputs/action4_fast_pipeline_repair_v001');self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'fixture.json';self.path.write_text('{"x":[1]}');self.ref={'path':str(self.path.relative_to(cache.R)),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}
    def test_bytes_once_and_seal(self):
        i=cache._ReadOnceIndex();self.assertEqual(i.read_ref(self.ref),{'x':[1]});i.read_ref(self.ref);i.check_ref(self.ref);self.assertEqual(i.events['reference_byte_reads'],1);i.seal()
        with self.assertRaises(ValueError):i.read_ref(self.ref)
    def test_conflicting_sha_rejected(self):
        i=cache._ReadOnceIndex();i.read_ref(self.ref)
        with self.assertRaises(ValueError):i.check_ref(dict(self.ref,sha256='0'*64))
    def test_changed_cached_document_rejected_at_seal(self):
        i=cache._ReadOnceIndex();i.read_ref(self.ref)['x'].append(2)
        with self.assertRaises(ValueError):i.seal()
    def test_fresh_byte_drift_rejected(self):
        i=cache._ReadOnceIndex();i.read_ref(self.ref);self.path.write_text('{"x":[2]}')
        with self.assertRaises(ValueError):cache.b._current_reference(self.ref)
    def test_wrong_active_plan_never_constructs(self):
        with patch.object(cache,'_ScopedPlanContext') as constructor:
            with self.assertRaises(ValueError):cache.approved_context(self.ref,object(),handoff_ref=self.ref)
            constructor.assert_not_called()
    def test_preparation_phase_never_constructs(self):
        with patch.object(cache.authority,'active',return_value=({'status':'PREPARATION_ONLY','phase':'preparation'},[])),patch.object(cache,'_ScopedPlanContext') as constructor:
            with self.assertRaises(ValueError):cache.approved_context(self.ref,object(),handoff_ref=self.ref)
            constructor.assert_not_called()
    def test_original_record_and_publication_methods_inherited(self):
        self.assertIs(cache.FastHostSession.validate_host_record,cache.b.HostValidationSession.validate_host_record)
        self.assertIs(cache.FastHostSession.close,cache.b.HostValidationSession.close)
        self.assertIs(cache.FastRenderBatch.render_entry,cache.b.RenderBatch.render_entry)
        self.assertIs(cache.FastRenderBatch.close,cache.b.RenderBatch.close)
if __name__=='__main__':unittest.main()
