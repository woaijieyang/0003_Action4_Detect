"""Boundary tests use explicit mocked checkpoint metadata, never accepted model results."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.action4_runner import freeze_test_selection, run_evaluation, tune_decode, _require_formal_checkpoint
from model.training.action4_runtime import sha256_file


class DatasetReached(RuntimeError):
    """Stop before dataset/model/scoring; these tests assert admission only."""


class EvalLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ckpt = self.root / 'NOT_A_MODEL.bin'
        self.ckpt.write_bytes(b'Unit fixture: checkpoint loader mocked; no usable weights')
        self.paths = {}
        for split in ('eval', 'test'):
            path = self.root / (split + '_manifest_sentinel.txt')
            path.write_text(split + ' sentinel; not an accepted dataset')
            self.paths[split] = path
        self.config = {'network': {}, 'preprocess': {}, 'decode': {'score_threshold': .3},
                       'runtime': {'device': 'cpu'},
                       'data': {k + '_manifest': str(p) for k,p in self.paths.items()}}
        self.state = {'claim_scope': 'accepted_data_model_training', 'stage': 'B0',
                      'epoch_boundary': True, 'global_step': 1,
                      'identity': {'data_sha256': {k: sha256_file(p) for k,p in self.paths.items()}}}
        patcher = mock.patch('eval.action4_runner.load_checkpoint', side_effect=lambda _: copy.deepcopy(self.state))
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = mock.patch('data.action4_dataset.Action4Dataset', side_effect=DatasetReached('admitted, stop before data/model'))
        self.dataset = patcher.start(); self.addCleanup(patcher.stop)
        patcher = mock.patch('eval.action4_runner.build_model', side_effect=AssertionError('model must never run in guard test'))
        patcher.start(); self.addCleanup(patcher.stop)

    def _invoke(self, entry, *, diagnostic=False):
        if entry == 'freeze':
            with mock.patch('eval.action4_runner.write_json_new') as writer:
                value = freeze_test_selection(self.root, self.config, self.ckpt, self.root/'never_written_selection.json')
            return value
        if entry == 'tune':
            return tune_decode(self.root, self.config, self.ckpt, {'score_threshold': [.2,.3]},
                               run_id='fixture_tune', allow_diagnostic=diagnostic)
        return run_evaluation(self.root, self.config, self.ckpt, split=entry,
                              run_id='fixture_'+entry, selection='missing_selection.json', allow_diagnostic=diagnostic)

    def test_replaced_fixed_manifest_rejected_before_dataset_and_output(self):
        for entry in ('freeze','test','eval','tune'):
            with self.subTest(entry=entry):
                split = 'test' if entry in ('freeze','test') else 'eval'
                original = self.paths[split].read_bytes()
                self.paths[split].write_bytes(original+b' changed')
                try:
                    with self.assertRaisesRegex(ValueError, 'frozen .* manifest'):
                        self._invoke(entry)
                    self.dataset.assert_not_called()
                    self.assertFalse((self.root/'outputs').exists())
                finally:
                    self.paths[split].write_bytes(original)

    def test_missing_or_malformed_lineage_is_not_silently_accepted(self):
        for entry in ('freeze','test','eval','tune'):
            for identity in ({}, {'data_sha256':{}}, {'data_sha256':None}, {'data_sha256':{'eval':'x','test':'x'}}):
                with self.subTest(entry=entry,identity=identity):
                    self.state['identity'] = identity
                    with self.assertRaisesRegex(ValueError, 'frozen .* manifest'):
                        self._invoke(entry)
        self.dataset.assert_not_called()
        self.assertFalse((self.root/'outputs').exists())

    def test_component_diagnostic_does_not_bypass_manifest_binding(self):
        self.state['claim_scope']='component_diagnostic'; self.state['stage']='D0'
        for entry in ('eval','tune'):
            with self.subTest(entry=entry):
                self.state['identity']['data_sha256']['eval']='0'*64
                with self.assertRaisesRegex(ValueError, 'frozen eval manifest'):
                    self._invoke(entry,diagnostic=True)
        self.dataset.assert_not_called()

    def test_matching_lineage_allows_eval_diagnostic_and_decode_only_tuning(self):
        self.config['decode']['score_threshold']=.7
        for scope in ('accepted_data_model_training','component_diagnostic'):
            self.state['claim_scope']=scope
            for entry in ('eval','tune'):
                with self.subTest(scope=scope,entry=entry):
                    with self.assertRaises(DatasetReached):
                        self._invoke(entry,diagnostic=scope=='component_diagnostic')
                    if (self.root/'outputs/action4/fixture_eval').exists():
                        (self.root/'outputs/action4/fixture_eval').rmdir()

    def test_matching_formal_freeze_allows_decode_change_without_writing_selection(self):
        self.config['decode']['score_threshold']=.7
        result=self._invoke('freeze')
        self.assertEqual(result['test_manifest_sha256'],self.state['identity']['data_sha256']['test'])
        self.assertEqual(result['inference_contract']['decode']['score_threshold'],.7)
        self.assertFalse((self.root/'never_written_selection.json').exists())
        self.dataset.assert_not_called()

    def test_matching_frozen_test_admitted_but_later_decode_change_rejected(self):
        selection = self.root/'fixture_selection.json'
        frozen=self._invoke('freeze')
        selection.write_text(json.dumps(frozen))
        with self.assertRaises(DatasetReached):
            run_evaluation(self.root,self.config,self.ckpt,split='test',run_id='fixture_valid_test',selection=selection)
        self.config['decode']['score_threshold']=.8
        with self.assertRaisesRegex(ValueError,'frozen model/config/data/code identity'):
            run_evaluation(self.root,self.config,self.ckpt,split='test',run_id='fixture_changed_decode',selection=selection)
        self.assertFalse((self.root/'outputs/action4/fixture_changed_decode').exists())

    def test_strict_completed_epoch_and_step_types(self):
        for boundary in (False,1,'true',None):
            with self.subTest(boundary=boundary):
                with self.assertRaises(ValueError):
                    _require_formal_checkpoint({**self.state,'epoch_boundary':boundary})
        for step in (None,False,True,0,-1,1.5,float('nan'),float('inf'),'1'):
            with self.subTest(step=repr(step)):
                with self.assertRaises(ValueError):
                    _require_formal_checkpoint({**self.state,'global_step':step})
        _require_formal_checkpoint(self.state)
        _require_formal_checkpoint({**self.state,'stage':'C1','global_step':10})

    def test_component_and_a0_cannot_freeze_formal_test(self):
        for scope,stage in (('component_diagnostic','B0'),('accepted_data_model_training','A0')):
            self.state.update(claim_scope=scope,stage=stage)
            with self.assertRaisesRegex(ValueError,'completed B0/C1'):
                self._invoke('freeze')
        self.dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
