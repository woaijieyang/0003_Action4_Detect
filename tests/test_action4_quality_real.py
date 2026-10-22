"""NOT RUN. Tiny real-record tests, require explicit post-128 GPU3 authorization.
ACTION4_QUALITY_TEST_MANIFESTS is a JSON list of exact existing shard manifests.
Reads originals only; does not write image files or change any source labels.
"""
from copy import deepcopy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/zy/vln/a-test')
SELF_REF={'path':str((ROOT/'src/eval/action4_gpu_quality.py').relative_to(ROOT)), 'sha256':hashlib.sha256((ROOT/'src/eval/action4_gpu_quality.py').read_bytes()).hexdigest()}
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('gpu_quality_real_candidate',ROOT/'src/eval/action4_gpu_quality.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RealRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.aid=m.GPUAid(implementation_ref=SELF_REF)
        cls.original=m.load_original()
        paths=json.loads(os.environ['ACTION4_QUALITY_TEST_MANIFESTS'])
        cls.samples={}
        for value in paths:
            manifest=cls.original.bounded(value)
            for line in manifest.read_text().splitlines():
                row=json.loads(line)
                count=len(row['instances'])
                if count in (0,2) and count not in cls.samples:
                    plan_ref=row['pose_controls_proof']['plan_ref']
                    reader=cls.original.ReadSet();plan=reader.json(plan_ref['path'],plan_ref['sha256']);reader.close()
                    entry=next(e for e in plan['entries'] if e['sample_id']==row['sample_id'])
                    cls.original.check_presence(row,entry)
                    cls.samples[count]=(row,manifest.parent,entry)
        if set(cls.samples)!={0,2}:
            raise ValueError('Explicit real fixture manifests must supply one zero and one dual record')

    def validate(self,row,data,gpu=True):
        reader=self.original.ReadSet()
        value=(self.aid if gpu else self.original).validate_record(reader,row,data)
        reader.close()
        return value

    def test_real_cpu_gpu_same_masks_points_and_support(self):
        row,data,_=self.samples[2]
        cpu=self.validate(row,data,gpu=False)
        gpu=self.validate(row,data,gpu=True)
        self.assertEqual(cpu[2],gpu[2])
        self.assertEqual(len(cpu[1]),len(gpu[1]))
        for a,b in zip(cpu[1],gpu[1]):
            self.assertEqual(a['instance_id'],b['instance_id'])
            self.assertEqual(a['outside_hull_pixels'],b['outside_hull_pixels'])
            self.assertEqual(a['point_inframe'],b['point_inframe'])
            self.assertEqual(a['visible_fraction'],b['visible_fraction'])
            self.assertEqual(a['visible_area_fullinput_px'],b['visible_area_fullinput_px'])
            np.testing.assert_allclose(a['independent_points_native_px'],b['independent_points_native_px'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(a['depth_m'],b['depth_m'],rtol=0,atol=1e-12)

    def test_wrong_canonical_point_order_rejected(self):
        row,data,_=self.samples[2];bad=deepcopy(row)
        bad['instances'][0]['canonical8_geometry']['corner_ids'][0:2]=[1,0]
        for gpu in (False,True):
            with self.assertRaises(ValueError):self.validate(bad,data,gpu=gpu)

    def test_shifted_stored_projection_rejected(self):
        row,data,_=self.samples[2];bad=deepcopy(row)
        bad['instances'][0]['canonical8_geometry']['points_native_px'][0][0]+=1.0
        for gpu in (False,True):
            with self.assertRaises(ValueError):self.validate(bad,data,gpu=gpu)

    def test_actual_zero_target_keeps_no_fake_gpu_projection(self):
        row,data,_=self.samples[0]
        before=self.aid.receipt()
        _,rows,semantic=self.validate(row,data)
        after=self.aid.receipt()
        self.assertEqual(rows,[])
        self.assertFalse(semantic['measurements_by_instance_id'])
        self.assertEqual(after['projection_calls'],before['projection_calls'])
        self.assertEqual(after['hull_pixels'],before['hull_pixels'])

    def test_zero_dummy_target_requests_replacement_same_entry(self):
        row,data,entry=self.samples[0];bad=deepcopy(row)
        bad['instances']=[{'instance_id':1}]
        request=m.check_or_request(self.aid,self.original.ReadSet(),bad,data,entry=entry,
                                  manifest_ref={'path':'test_fixture_only','sha256':'0'*64})
        self.assertEqual(request['status'],'REJECT_REVIEW_AND_REPLACEMENT_REQUEST')
        self.assertEqual(request['expected_entry'],entry)
        self.assertEqual(request['required_same_split'],entry['split'])
        self.assertEqual(request['required_same_group_id'],entry['group_id'])
        self.assertFalse(request['automatic_acceptance'])
        self.assertFalse(request['training_accepted'])


if __name__=='__main__':unittest.main()
