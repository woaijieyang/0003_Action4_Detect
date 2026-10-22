"""CPU checks for the controlled skin-colour ablation, never GPU execution."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
from data import action4_skin_diagnostic as m

class SkinDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((m.ROOT / 'configs/data/action4_skin_diagnostic_v001.json').read_text())

    def test_linearize_before_average(self):
        rgb = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(m.mean_linear_rgb(rgb), [.5, .5, .5])
        self.assertGreater(m.mean_linear_rgb(rgb)[0], ((.5 + .055) / 1.055) ** 2.4)

    def test_piecewise_transfer_at_byte_boundary(self):
        rgb = np.array([[[10, 11, 255]]], dtype=np.uint8)
        np.testing.assert_allclose(m.mean_linear_rgb(rgb), [(10 / 255) / 12.92, ((11 / 255 + .055) / 1.055) ** 2.4, 1.], atol=1e-15)

    def test_reject_implicit_source_conversions(self):
        for rgb in (np.zeros((1, 1, 4), np.uint8), np.zeros((1, 1, 3), np.uint16), np.zeros((1, 1, 3), float), np.zeros((0, 1, 3), np.uint8)):
            with self.assertRaises(ValueError): m.mean_linear_rgb(rgb)

    def test_actual_mean_and_full_preview_contract(self):
        with patch.object(m.renderer, 'validate_config', wraps=m.renderer.validate_config) as original:
            preview, contract, shot, evidence = m.validate_config(self.cfg, recompute_mean=True)
            self.assertEqual(len(original.call_args.args[0]['inspection_poses']), 3)
        self.assertEqual(shot['id'], 'left_wrist')
        self.assertEqual(evidence['pixel_count'], 25600)
        self.assertEqual(contract['config']['sources']['skin_forearm'], evidence['source'])
        self.assertEqual(preview['render']['denoiser'], 'OPTIX')

    def test_mean_recompute_mismatch_rejects(self):
        value = m.derive_mean(self.cfg['sources']['photo']['path'])
        value['linear_rgb_mean_float64'][0] += .01
        with patch.object(m, 'derive_mean', return_value=value):
            with self.assertRaisesRegex(ValueError, 'recomputation'): m.validate_config(self.cfg, recompute_mean=True)

    def test_fixed_ablation_scope(self):
        for key, value in [('shot_id', 'whole_table'), ('mix_factor', .5), ('variants', ['constant_linear_mean']), ('threads', 1), ('dataset_generation_allowed', True)]:
            cfg = copy.deepcopy(self.cfg); cfg[key] = value
            with self.assertRaises(ValueError): m.validate_config(cfg)

    def test_source_hash_mismatch(self):
        cfg = copy.deepcopy(self.cfg); cfg['sources']['photo']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): m.validate_config(cfg)

    def test_wrong_resource_before_worker_query(self):
        for gpu, threads, probe in [('GPU-invalid', 2, False), (None, 1, True), (m.GPU_UUID, 2, True)]:
            with patch.object(m.renderer, 'worker_environment') as worker:
                with self.assertRaises(ValueError):
                    m.launch('configs/data/action4_skin_diagnostic_v001.json', m.OUTPUTS / 'not_created_by_unit_test', threads, gpu_uuid=gpu, probe_only=probe)
                worker.assert_not_called()
        self.assertFalse((m.OUTPUTS / 'not_created_by_unit_test').exists())

    def test_owned_fresh_outputs_only(self):
        for value in ('data/action4_synth/skin', 'outputs/another_audit/skin', m.OUTPUTS, '../outside'):
            with self.assertRaises(ValueError): m.output_path(value)
        with self.assertRaises(FileExistsError): m.output_path(m.OUTPUTS / 'preflight_v001')

if __name__ == '__main__': unittest.main(verbosity=2)
