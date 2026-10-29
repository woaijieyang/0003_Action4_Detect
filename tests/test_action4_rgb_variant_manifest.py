"""Pixel-only engineering fixtures; no SAM, detector, labels or training."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from data.action4_paths import PROJECT_ROOT as R
from data.action4_rgb_variant_manifest import materialize, verify_manifest
from data.action4_rgb_variants import DEFAULT_POLICY

OUT=R/"outputs/action4_rgb_variant_manifest_preparation_v001"
def bind(path):
    return {"path":str(path.relative_to(R)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

class ManifestTests(unittest.TestCase):
    def setUp(self):
        OUT.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=OUT);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.rgb=np.arange(12*13*3,dtype=np.uint16).reshape(12,13,3).astype(np.uint8)
        self.source=self.root/"source.png";Image.fromarray(self.rgb).save(self.source)
        row={"sample_id":"image-A","split":"train","status":"training_accepted","image":bind(self.source)["path"],
             "rgb_sha256":bind(self.source)["sha256"],"height":12,"width":13,
             "instances":[{"mask_visible":"DO_NOT_COPY","pose":"DO_NOT_COPY"}]}
        self.annotations=self.root/"annotations.jsonl";self.annotations.write_text(json.dumps(row)+"\n")
        self.policy=self.root/"policy.json";self.policy.write_text(json.dumps(DEFAULT_POLICY))
        self.args=(bind(self.annotations),bind(self.policy),["image-A"])
    def make(self,name="pilot"):
        return materialize(*self.args,self.root/name,seed=8)
    def test_complete_identity_and_original_not_copied(self):
        binding=self.make();m=verify_manifest(binding)
        self.assertEqual(len(m["rows"]),3);self.assertEqual(m["scope"]["requested_images"],1)
        self.assertEqual(m["scope"]["dataset_units_added"],0)
        self.assertFalse(m["scope"]["formal_stage3_admitted"])
        self.assertEqual(m["rows"][0]["actual_input"],bind(self.source))
        self.assertEqual(m["rows"][0]["legacy_leaf_key"],"native_original")
        self.assertTrue(m["rows"][0]["recipe"]["variant_id"].startswith("native_original:"))
        self.assertFalse(m["rows"][0]["legacy_alias_admitted"])
        self.assertNotIn("DO_NOT_COPY",json.dumps(m))
        for row in m["rows"]:
            self.assertEqual(row["image_id"],"image-A")
            self.assertEqual(row["native_shape"],[12,13,3])
            self.assertEqual(row["original_source"],bind(self.source))
        self.assertEqual(len(list((self.root/"pilot").glob("*.png"))),2)
        with self.assertRaises(FileExistsError):self.make()
    def test_tampered_derived_file(self):
        binding=self.make();m=verify_manifest(binding)
        p=R/m["rows"][1]["actual_input"]["path"];Image.fromarray(np.zeros_like(self.rgb)).save(p)
        with self.assertRaises(ValueError):verify_manifest(binding)
    def test_source_and_policy_tampering(self):
        binding=self.make()
        self.source.write_bytes(b"changed source")
        with self.assertRaises(ValueError):verify_manifest(binding)
        with self.assertRaises(ValueError):self.make("changed")
        self.assertFalse((self.root/"changed"/"READY.json").exists())
        self.policy.write_text("{}")
        with self.assertRaises(ValueError):materialize(*self.args,self.root/"badpolicy")
    def test_missing_duplicate_and_annotation_hash(self):
        for ids in ([],["absent"],["image-A","image-A"]):
            with self.assertRaises(ValueError):materialize(self.args[0],self.args[1],ids,self.root/"never")
        original=json.loads(self.annotations.read_text())
        self.annotations.write_text(json.dumps(original)+"\n"+json.dumps(original)+"\n")
        with self.assertRaises(ValueError):materialize(bind(self.annotations),self.args[1],["image-A"],self.root/"duplicate")
        with self.assertRaises(ValueError):materialize(*self.args,self.root/"tampered")
    def test_partial_png_write_never_ready(self):
        real=Image.Image.save
        def fail(image,stream,*args,**kwargs):
            stream.write(b"partial");raise OSError("injected partial write")
        with patch.object(Image.Image,"save",fail):
            with self.assertRaisesRegex(OSError,"partial"):self.make()
        out=self.root/"pilot"
        self.assertTrue((out/"EXPECTED.json").exists());self.assertTrue((out/"FAILED.json").exists())
        self.assertFalse((out/"MANIFEST.json").exists());self.assertFalse((out/"READY.json").exists())
        with self.assertRaises(FileExistsError):self.make()
    def test_same_recipe_independent_outputs_and_geometry_rejection(self):
        a=verify_manifest(self.make("a"));b=verify_manifest(self.make("b"))
        for x,y in zip(a["rows"],b["rows"]):
            self.assertEqual(x["recipe"],y["recipe"]);self.assertEqual(x["actual_pixels_sha256"],y["actual_pixels_sha256"])
        row=json.loads(self.annotations.read_text());row["width"]=99
        self.annotations.write_text(json.dumps(row)+"\n")
        with self.assertRaises(ValueError):materialize(bind(self.annotations),self.args[1],["image-A"],self.root/"badshape")
        self.assertFalse((self.root/"badshape"/"READY.json").exists())
    def test_rebound_incomplete_manifest_and_rehashed_wrong_pixels(self):
        binding=self.make();m=verify_manifest(binding)
        path=R/binding["path"]
        m["rows"].pop();path.write_text(json.dumps(m))
        altered=bind(path)
        (path.parent/"READY.json").write_text(json.dumps({"status":"READY_PIXEL_PILOT_ONLY","manifest":altered,"expected_registration":m["expected_registration"]}))
        with self.assertRaises(ValueError):verify_manifest(altered)
        binding=self.make("other");m=verify_manifest(binding);path=R/binding["path"]
        image=R/m["rows"][1]["actual_input"]["path"]
        Image.fromarray(np.zeros_like(self.rgb)).save(image)
        m["rows"][1]["actual_input"]=bind(image)
        path.write_text(json.dumps(m));altered=bind(path)
        (path.parent/"READY.json").write_text(json.dumps({"status":"READY_PIXEL_PILOT_ONLY","manifest":altered,"expected_registration":m["expected_registration"]}))
        with self.assertRaises(ValueError):verify_manifest(altered)
    def test_resolved_source_held_and_policy_binding_strict(self):
        import data.action4_rgb_variant_manifest as module
        alias=self.root/"alias.png";alias.symlink_to(self.source)
        other=self.root/"other.png";other.write_bytes(self.source.read_bytes())
        record=json.loads(self.annotations.read_text());record["image"]=str(alias.relative_to(R))
        real=module.generate_from_file
        def change_alias(path,*args,**kwargs):
            self.assertEqual(Path(path),self.source)
            alias.unlink();alias.symlink_to(other)
            return real(path,*args,**kwargs)
        with patch.object(module,"generate_from_file",change_alias):
            _,_,receipt=module._recipe(record,"original",8,DEFAULT_POLICY)
        self.assertEqual(receipt["path"],str(self.source.relative_to(R)))
        bad=dict(self.args[1],GT_pose="must not copy")
        with self.assertRaises(ValueError):materialize(self.args[0],bad,self.args[2],self.root/"badbinding")
    def test_resealed_domain_cannot_shrink_preregistered_expected(self):
        a=json.loads(self.annotations.read_text());b=copy.deepcopy(a);b["sample_id"]="image-B"
        self.annotations.write_text(json.dumps(a)+"\n"+json.dumps(b)+"\n")
        binding=materialize(bind(self.annotations),self.args[1],["image-A","image-B"],self.root/"two",seed=8)
        path=R/binding["path"];manifest=json.loads(path.read_text())
        manifest["image_ids"]=["image-A"]
        manifest["expected"]=[x for x in manifest["expected"] if x["image_id"]=="image-A"]
        manifest["rows"]=[x for x in manifest["rows"] if x["image_id"]=="image-A"]
        manifest["scope"].update(requested_images=1,variant_rows=3)
        path.write_text(json.dumps(manifest));altered=bind(path)
        ready=json.loads((path.parent/"READY.json").read_text());ready["manifest"]=altered
        (path.parent/"READY.json").write_text(json.dumps(ready))
        with self.assertRaisesRegex(ValueError,"predeclared"):
            verify_manifest(altered)

    def test_paths_and_ready_binding(self):
        with self.assertRaises(ValueError):materialize(*self.args,"../escape")
        binding=self.make();ready=self.root/"pilot"/"READY.json"
        ready.write_text(json.dumps({"status":"READY_PIXEL_PILOT_ONLY","manifest":{"path":"wrong","sha256":"a"*64}}))
        with self.assertRaises(ValueError):verify_manifest(binding)

if __name__=="__main__":unittest.main()
