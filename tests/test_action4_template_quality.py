"""CPU-only tests; real diagnostics never masquerade as a batch release."""
import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path('/data/zy/vln/a-test')
sys.path.insert(0, str(ROOT/'src'))
from eval.action4_template_quality import (IntegrityError, validate_presence,
    projected_geometry, result_status, _engine, require_integrity, Inputs, track_record_pngs, file_sha)
from eval.action4_gpu_quality import FP64Math


class TemplateQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.host = json.loads((ROOT/'outputs/action4_diverse_sampling_v007/fixed_scene_rgb_layout_v006/HOST_CONSUMER_REPORT.json').read_text())
        cls.fixtures = []
        mapping = {'wall':'wall', 'table_full_garment':'table_v002', 'cloth_lift10_full_garment':'cloth_v002'}
        for name, subdir in mapping.items():
            data = ROOT/'outputs/action4_diverse_sampling_v007/fixed_scene_rgb_layout_v006'/name
            record = json.loads((data/'record_with_template.json').read_text())
            meta = json.loads((ROOT/'data/action4_scene_templates/v007_candidate'/subdir/'template.json').read_text())
            with (ROOT/'outputs/action4_diverse_sampling_v007/fixed_scene_v005'/name/'parameters.jsonl').open() as f:
                proposal = json.loads(next(f))
            source = json.loads((ROOT/meta['canonical_definition']['source_manifest']['path']).read_text())
            semantics = next(x['host']['mask_semantics'] for x in cls.host['samples'] if x['template'] == name)
            cls.fixtures.append((record, proposal, meta, source, semantics, data))

    def geometry(self, record=None):
        original, proposal, meta, source, semantics, data = self.fixtures[0]
        return projected_geometry(record or original, proposal, meta, source,
                                  FP64Math('cpu'), semantics, data)

    def test_three_actual_diagnostic_projection_and_mask_hulls_cpu_only(self):
        # Actual PNG pixels and prior host semantics; kernel regression only.
        count = 0
        for record, proposal, meta, source, semantics, data in self.fixtures:
            engine = FP64Math('cpu')
            result = projected_geometry(record, proposal, meta, source, engine, semantics, data)
            self.assertEqual(len(result), 2)
            self.assertEqual(engine.projection_calls, 2)
            self.assertEqual(engine.hull_calls, 4)
            self.assertGreater(engine.hull_pixels, 0)
            count += sum(len(r['independent_points_native_px']) for r in result)
        self.assertEqual(count, 48)

    def test_forged_canonical_source_and_projection_rejected(self):
        row = copy.deepcopy(self.fixtures[0][0])
        row['instances'][0]['canonical8_geometry']['points_object_mm'][0][0] += 1
        with self.assertRaises(ValueError):
            self.geometry(row)
        row = copy.deepcopy(self.fixtures[0][0])
        row['instances'][0]['canonical8_geometry']['points_native_px'][0][0] += 10
        with self.assertRaises(ValueError):
            self.geometry(row)

    def test_nonfinite_labels_rejected(self):
        for value in (float('nan'), float('inf')):
            row = copy.deepcopy(self.fixtures[0][0])
            row['instances'][0]['corners_px'][0][0] = value
            with self.assertRaises(ValueError):
                self.geometry(row)

    def test_retained_absent_identity_cannot_become_label(self):
        row = copy.deepcopy(self.fixtures[0][0])
        entry = dict(self.fixtures[0][1]['entry'])
        row['status'] = 'render_candidate_not_training_accepted'
        entry.update(target_presence=[False, True], target_count=1, split='train')
        row.update(entry, planned_target_count=1)
        with self.assertRaisesRegex(ValueError, 'instance'):
            validate_presence(row, entry)
        row['instances'] = [row['instances'][1]]
        validate_presence(row, entry)
        entry.update(target_presence=[False, False], target_count=0)
        row.update(entry, planned_target_count=0, instances=[])
        validate_presence(row, entry)
        row['instances'] = [copy.deepcopy(self.fixtures[0][0]['instances'][0])]
        with self.assertRaises(ValueError):
            validate_presence(row, entry)

    def test_test_split_is_dual_and_accepted_records_forbidden(self):
        row = copy.deepcopy(self.fixtures[0][0])
        entry = dict(self.fixtures[0][1]['entry'])
        row.update(status='training_accepted', training_accepted=True)
        with self.assertRaises(ValueError):
            validate_presence(row, entry)
        entry.update(split='test', target_presence=[False, True], target_count=1)
        row.update(entry, planned_target_count=1, instances=[row['instances'][1]],
                   status='render_candidate_not_training_accepted', training_accepted=False)
        with self.assertRaisesRegex(ValueError, 'dual'):
            validate_presence(row, entry)

    def test_cpu_never_production_pass_and_cuda_failure_no_fallback(self):
        self.assertEqual(result_status([], [], test_cpu_math=True),
                         'TEST_ONLY_CPU_MATH_NO_PRODUCTION_ACCEPTANCE')
        self.assertIn('REJECT', result_status(['bad'], [], test_cpu_math=False))
        self.assertIn('REJECT', result_status([], ['missing'], test_cpu_math=False))
        with patch('eval.action4_gpu_quality.GPUAid', side_effect=RuntimeError('CUDA unavailable')):
            with patch('eval.action4_gpu_quality.FP64Math') as fallback:
                with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                    _engine(False)
                fallback.assert_not_called()

    def test_identity_failure_and_midcheck_drift_are_operational(self):
        base = ROOT/'outputs/action4_diverse_sampling_v007/template_quality_v001/test_fixtures'
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            path = Path(directory)/'source.json'
            path.write_text('{"version":1}')
            with self.assertRaises(IntegrityError):
                Inputs().read({'path':str(path.relative_to(ROOT)), 'sha256':'0'*64})
            inputs = Inputs()
            self.assertEqual(inputs.current(path), {'version':1})
            path.write_text('{"version":2}')
            with self.assertRaises(IntegrityError):
                inputs.close()
        self.assertFalse(issubclass(IntegrityError, ValueError))

    def test_png_initial_mismatch_and_post_host_drift_fail_batch_integrity(self):
        from PIL import Image
        base = ROOT/'outputs/action4_diverse_sampling_v007/template_quality_v001/test_fixtures'
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            paths = [Path(directory)/(name+'.png') for name in ('rgb','visible','amodal')]
            for path in paths:
                Image.new('RGB', (2,2), (12,34,56)).save(path)
            refs = [str(path.relative_to(ROOT)) for path in paths]
            hashes = [file_sha(path) for path in paths]
            record = {'image':refs[0], 'rgb_sha256':hashes[0], 'excluded_instances':[],
                      'instances':[{'mask_visible':refs[1], 'mask_visible_sha256':hashes[1],
                                    'mask_amodal_expanded':refs[2], 'mask_amodal_expanded_sha256':hashes[2]}]}
            bad = copy.deepcopy(record)
            bad['rgb_sha256'] = '0'*64
            with self.assertRaises(IntegrityError):
                track_record_pngs(Inputs(), bad)
            for position, path in enumerate(paths):
                with self.subTest(consumed_png=position):
                    inputs = Inputs()
                    track_record_pngs(inputs, record)
                    self.assertEqual(set(inputs.refs), set(refs))
                    # Simulates mutation after successful host validation and before close.
                    Image.new('RGB', (2,2), (98,76,54)).save(path)
                    with self.assertRaises(IntegrityError):
                        inputs.close()
                    Image.new('RGB', (2,2), (12,34,56)).save(path)

    def test_actual_dataset_consumers_and_formal_rejection_cpu_only(self):
        from data.action4_dataset import Action4Dataset
        from eval.action4_synthesis_validation import consumer_check
        config = json.loads((ROOT/'configs/model/action4_pilot2000_v002.json').read_text())
        base = ROOT/'outputs/action4_diverse_sampling_v007/template_quality_v001/test_fixtures'
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            for n, fixture in enumerate(self.fixtures):
                row = fixture[0]
                path = Path(directory)/('diagnostic_'+str(n)+'.jsonl')
                path.write_text(json.dumps(row)+chr(10))
                ds = Action4Dataset(ROOT, path, allow_diagnostic=True, split=row['split'])
                self.assertEqual(ds.records, [row])
                with self.assertRaisesRegex(ValueError, 'unaccepted'):
                    Action4Dataset(ROOT, path)
                result = consumer_check(ds[0], config)
                self.assertEqual(result['physical_ids'], [1,2])
                self.assertEqual(result['supported_corner_count'],16)
                self.assertLess(result['max_native_round_trip_error_px'],1e-6)


if __name__ == '__main__':
    unittest.main()
