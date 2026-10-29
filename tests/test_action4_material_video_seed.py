"""CPU video identity checks on existing frame4; no model or tracker calls."""
import copy,importlib.util,json,pickle,unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from data.action4_sam_roi_inputs import ImageFrame
from data.action4_video_material_support import VerifiedVideoMaterialSupport
ROOT=Path("/data/zy/vln/a-test")

class VideoMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location("material_cpu_inputs",ROOT/"outputs/action4_material_video_seed_preparation_v001/verify_real.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.args=module.inputs()
        cls.support=VerifiedVideoMaterialSupport.from_current_response(**cls.args)
    def consume(self,**changes):
        a=self.args
        kwargs=dict(expected_models=a["expected_models"],expected_sources=self.support.sources,track_id="track_00",
                    track_generation=0,frame_id=4,pts=a["frame_binding"].pts,time_base=a["frame_binding"].time_base)
        kwargs.update(changes)
        return self.support.consume(a["surface"],a["frame"],**kwargs)
    def test_actual_current_response_and_namespace_timing(self):
        mask,evidence=self.consume()
        self.assertTrue(mask.any());self.assertFalse(evidence["geometry_accepted"])
        self.assertEqual(evidence["frame_id"],4);self.assertEqual(evidence["native_obj_id"],0)
        self.assertEqual(evidence["support_semantics"],"current_predicted_visible_body_not_full_body_GT")
        self.assertNotEqual(self.args["frame"].image_id,self.args["source_frame"].image_id)
        self.assertFalse(evidence["physical_or_semantic_identity_verified"])
    def test_wrong_frame_pts_instance_generation_model(self):
        for changes in (dict(frame_id=5),dict(pts=self.args["frame_binding"].pts+1),
                        dict(time_base="1/30"),dict(track_id="track_01"),dict(track_generation=1)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.consume(**changes)
        wrong=copy.deepcopy(self.args["expected_models"]);wrong["A0"]["sha256"]="f"*64
        with self.assertRaises(ValueError):self.consume(expected_models=wrong)
        with self.assertRaises(ValueError):self.consume(expected_sources={"b0_model_sha256":"f"*64})
    def test_capture_rejects_RGB_and_foreign_current_identity(self):
        args=dict(self.args);old=args["frame"];pixels=old.pixels.copy();pixels[0,0,0]^=1
        args["frame"]=replace(old,pixels=pixels)
        with self.assertRaises(ValueError):VerifiedVideoMaterialSupport.from_current_response(**args)
        for key,value in (("track_id","track_01"),("track_generation",1)):
            args=dict(self.args);args[key]=value
            with self.assertRaises(ValueError):VerifiedVideoMaterialSupport.from_current_response(**args)
    def test_native_generation_and_false_context_rejected(self):
        args=dict(self.args)
        ledger=json.loads(args["context"].source_ledger_json)
        ledger["native_owners"]["track_00"]["generation"]+=1
        args["context"]=replace(args["context"],source_ledger_json=json.dumps(ledger))
        with self.assertRaises(ValueError):VerifiedVideoMaterialSupport.from_current_response(**args)
        args=dict(self.args);args["context"]=object()
        with self.assertRaises(ValueError):VerifiedVideoMaterialSupport.from_current_response(**args)
    def test_fake_evidence_and_direct_dataclass_not_admitted(self):
        args=dict(self.args);args["owner_ledger_ref"]=dict(args["owner_ledger_ref"],sha256="f"*64)
        with self.assertRaises(ValueError):VerifiedVideoMaterialSupport.from_current_response(**args)
        args=dict(self.args);args["native_status_ref"]=dict(args["native_status_ref"],sha256="f"*64)
        with self.assertRaises(ValueError):VerifiedVideoMaterialSupport.from_current_response(**args)
        fake=replace(self.support)
        a=self.args
        with self.assertRaises(ValueError):
            fake.consume(a["surface"],a["frame"],expected_models=a["expected_models"],expected_sources=fake.sources,
                         track_id="track_00",track_generation=0,frame_id=4,pts=a["frame_binding"].pts,time_base=a["frame_binding"].time_base)
        with self.assertRaises(TypeError):pickle.dumps(self.support)
    def test_no_eval_dependency_in_data_or_model_adapter(self):
        import ast
        for name in ("src/data/action4_video_material_support.py","src/model/action4_surface_anchors.py"):
            tree=ast.parse((ROOT/name).read_text())
            imports=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any(x=="eval" or x.startswith("eval.") for x in imports))

if __name__=="__main__":unittest.main()
