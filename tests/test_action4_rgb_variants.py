"""CPU engineering contract tests; no dataset integration, cache generation or models."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_rgb_variants import (DEFAULT_POLICY, generate_variant, generate_from_file,
    select_variant, verify_variant, assert_cache_compatible, load_policy)
from data.action4_sam_roi_inputs import array_digest

OUT = R / "outputs/action4_rgb_variants_preparation_v001"

class RGBVariantsTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.arange(17*19*3, dtype=np.uint16).reshape(17,19,3).astype(np.uint8)
        self.kw = dict(image_id="engineering_image", source_sha256="a"*64, seed=71)
    def make(self, variant, **kwargs):
        return generate_variant(self.rgb, variant=variant, **dict(self.kw, **kwargs))
    def test_original_exact_and_readonly(self):
        out, meta = self.make("original")
        self.assertTrue(np.array_equal(out, self.rgb))
        self.assertEqual(meta["variant_sha256"], array_digest(out))
        with self.assertRaises(ValueError):out.setflags(write=True)
        out_before = out.copy()
        self.rgb[:] = 0
        self.assertTrue(np.array_equal(out, out_before))
    def test_reorder_resume_epoch_and_instances(self):
        first = {image: [select_variant(image, e, seed=11) for e in range(1,81)] for image in ("A","B","C")}
        resumed = {image: [select_variant(image, e, seed=11) for e in range(21,81)] for image in ("C","A","B")}
        for image in first:
            self.assertEqual(first[image][20:], resumed[image])
            self.assertEqual(set(first[image]), {"original","light","blur"})
            for epoch in (1,8,37):
                # Both units use only image ID; instance enumeration cannot alter RGB.
                a = self.make(select_variant(image, epoch, seed=11), image_id=image)
                b = self.make(select_variant(image, epoch, seed=11), image_id=image)
                self.assertEqual(a[1], b[1])
                self.assertTrue(np.array_equal(a[0], b[0]))
    def test_source_policy_and_image_binding(self):
        _, original = self.make("light")
        _, changed_source = self.make("light", source_sha256="b"*64)
        _, changed_image = self.make("light", image_id="other")
        p = copy.deepcopy(DEFAULT_POLICY);p["add_range"]=[-.08,.08]
        _, changed_policy = self.make("light", policy=p)
        self.assertEqual(len({m["variant_id"] for m in [original,changed_source,changed_image,changed_policy]}),4)
        altered = self.rgb.copy();altered[0,0,0] ^= 1
        _, changed_pixels = generate_variant(altered, variant="light", **self.kw)
        self.assertNotEqual(changed_pixels["variant_id"], original["variant_id"])
    def test_rehashed_wrong_pixels_and_cache_reuse_rejected(self):
        out, meta = self.make("light")
        self.assertTrue(verify_variant(out,meta,source_rgb=self.rgb,variant="light",**self.kw))
        bad = out.copy();bad[0,0,0] ^= 1
        with self.assertRaises(ValueError):verify_variant(bad,meta,source_rgb=self.rgb,variant="light",**self.kw)
        forged=copy.deepcopy(meta);forged["variant_sha256"]=array_digest(bad)
        with self.assertRaises(ValueError):verify_variant(bad,forged,source_rgb=self.rgb,variant="light",**self.kw)
        assert_cache_compatible(meta,copy.deepcopy(meta))
        _, other = self.make("original")
        with self.assertRaises(ValueError):assert_cache_compatible(meta,other)
    def test_transform_ranges_and_tiny_blur(self):
        active=set()
        for seed in range(40):
            for variant in ("light","blur"):
                out, meta=self.make(variant,seed=seed)
                self.assertEqual(out.shape,self.rgb.shape);self.assertEqual(out.dtype,np.uint8)
                p=meta["parameters"]
                for name,value in p["values"].items():
                    bounds=DEFAULT_POLICY["blur_sigma_range" if name=="blur" else name+"_range"]
                    self.assertTrue(bounds[0]<=value<=bounds[1])
                    if p["enabled"] and p["operations"][name]:active.add(name)
                self.assertTrue(verify_variant(out,meta,source_rgb=self.rgb,variant=variant,**dict(self.kw,seed=seed)))
        self.assertEqual(active,{"add","multiply","contrast","blur"})
        tiny=np.full((1,1,3),127,np.uint8)
        for seed in range(10):
            out,_=generate_variant(tiny,variant="blur",**dict(self.kw,seed=seed))
            self.assertTrue(np.array_equal(out,tiny))
    def test_input_validation_and_policy(self):
        for rgb in (self.rgb.astype(float),np.full((2,2,3),np.nan),np.zeros((0,2,3),np.uint8),
                    np.zeros((2,2,4),np.uint8),[[[0,0,0]]]):
            with self.assertRaises(ValueError):generate_variant(rgb,variant="original",**self.kw)
        for epoch in (0,-1,True,1.5):
            with self.assertRaises(ValueError):select_variant("A",epoch)
        for seed in (True,-1,1.2):
            with self.assertRaises(ValueError):self.make("light",seed=seed)
        for key,value in (("probability",True),("blur_sigma_range",[0,2]),("add_range",[float("nan"),.09]),("extra",1)):
            p=copy.deepcopy(DEFAULT_POLICY);p[key]=value
            with self.assertRaises(ValueError):self.make("light",policy=p)
    def test_file_receipt_keeps_initial_resolved_source_on_alias_retarget(self):
        from unittest.mock import patch
        import data.action4_rgb_variants as module
        OUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as folder:
            folder=Path(folder);a=folder/"A.png";b=folder/"B.png";alias=folder/"source.png"
            Image.fromarray(self.rgb).save(a);Image.fromarray(self.rgb ^ 255).save(b)
            alias.symlink_to(a)
            sha=hashlib.sha256(a.read_bytes()).hexdigest()
            original=module.generate_variant
            def retarget(*args,**kwargs):
                alias.unlink();alias.symlink_to(b)
                return original(*args,**kwargs)
            with patch.object(module,"generate_variant",side_effect=retarget):
                out,metadata,receipt=module.generate_from_file(alias,sha,image_id="A",variant="original")
            self.assertTrue(np.array_equal(out,self.rgb))
            self.assertEqual(receipt["path"],str(a.relative_to(R)))
            self.assertEqual(hashlib.sha256((R/receipt["path"]).read_bytes()).hexdigest(),receipt["source_sha256"])

    def test_file_same_bytes_policy_and_path_boundaries(self):
        OUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as folder:
            p=Path(folder)/"source.png";Image.fromarray(self.rgb).save(p)
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            out,meta,receipt=generate_from_file(p,sha,image_id="engineering_image",variant="light",seed=71)
            expected,expected_meta=self.make("light",source_sha256=sha)
            self.assertTrue(np.array_equal(out,expected));self.assertEqual(meta,expected_meta)
            self.assertTrue(receipt["source_file_verified"])
            with self.assertRaises(ValueError):generate_from_file(p,"b"*64,image_id="A",variant="original")
            with self.assertRaises(ValueError):generate_from_file("../outside.png",sha,image_id="A",variant="original")
            p.write_bytes(b"replaced")
            with self.assertRaises(ValueError):generate_from_file(p,sha,image_id="A",variant="original")
        policy=R/"configs/data/action4_rgb_variants_v001.json"
        self.assertEqual(load_policy(policy,hashlib.sha256(policy.read_bytes()).hexdigest()),DEFAULT_POLICY)
        with self.assertRaises(ValueError):load_policy(policy,"a"*64)

if __name__=="__main__":unittest.main()
