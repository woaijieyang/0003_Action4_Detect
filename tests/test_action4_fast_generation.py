"""Fast scene lifecycle and CLI tests; fixtures never imply rendered acceptance."""
import unittest,threading,os
from types import SimpleNamespace
from unittest.mock import Mock,patch
from data import action4_persistent_base as base
from data import action4_fast_launcher as launcher
from data import action4_fast_generation as generation

class FastExecutionTests(unittest.TestCase):
    def test_initialization_failure_restores_owned_base(self):
        owned=SimpleNamespace(scene=object(),restore=Mock())
        with patch.object(base.wearing,'create',return_value=owned),patch.object(base.sampling,'actor_bindings',side_effect=ValueError('broken bindings')):
            with self.assertRaisesRegex(ValueError,'broken bindings'):base.PersistentWearingBase({'wearing_context':{}})
        owned.restore.assert_called_once()
    def test_failed_lease_still_attempts_original_restore(self):
        pool=object.__new__(base.PersistentWearingBase)
        pool.pid=os.getpid();pool.thread=threading.get_ident();pool.closed=False;pool.poisoned=False
        pool.active=SimpleNamespace(restore=Mock(side_effect=ValueError('lease failure')))
        pool.base=SimpleNamespace(restore=Mock(),restored=True)
        with self.assertRaisesRegex(RuntimeError,'original base restoration attempted'):pool.close()
        pool.base.restore.assert_called_once();self.assertTrue(pool.closed);self.assertTrue(pool.poisoned)
    def test_foreign_close_never_mutates(self):
        pool=object.__new__(base.PersistentWearingBase);pool.pid=-1;pool.thread=threading.get_ident()
        pool.base=SimpleNamespace(restore=Mock())
        with self.assertRaisesRegex(ValueError,'Same-owner'):pool.close()
        pool.base.restore.assert_not_called()
    def test_stale_scope_rejected(self):
        with self.assertRaises(ValueError):generation.FastEntryScope(object(),{},{},{},{})
    def test_inner_command_preserves_native_selection(self):
        args=launcher.command('/x/blender','p','o',3,1,36,'GPU-X',0,4,None)
        self.assertIn('_render-shard',args)
        self.assertEqual(args[args.index('--entry-count')+1],'4')
        self.assertEqual(args[args.index('--threads')+1],'3')
        self.assertTrue(args[args.index('--python')+1].endswith('scripts/action4_render_fast.py'))
if __name__=='__main__':unittest.main()
