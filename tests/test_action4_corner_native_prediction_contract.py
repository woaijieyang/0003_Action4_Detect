"""CPU contracts only: tiny declared fixtures, no model weights or training."""
import copy,hashlib,json,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from data.action4_paths import PROJECT_ROOT as R
from data.action4_prediction_roi_data import validate_config,SAM_NATIVE_PROTOCOL,CORNER_NATIVE_PROTOCOL
from model.training.action4_roi_adaptation import validate_parent_input_contract
from data.action4_sam_roi_inputs import ImageFrame,NativeMaskIdentity,native_key,array_digest
from eval.action4_sam_pipeline_inputs import SamInferenceContext
from eval.action4_corner_prediction import _corner_input_contract,_corner_candidates,_prepare_corner_sam

class CornerNativeContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((R/"runtime/action4_runs/p2_a0_sam_prediction15_integration28_v001/source/resolved_config.json").read_text())
        self.parent=copy.deepcopy(self.cfg)
        self.cfg.update(training_protocol=CORNER_NATIVE_PROTOCOL,a0_variant="a0_corner_image",corner_enabled=True)
        self.cfg["corner_counterpart"]={"B0":{"path":self.cfg["evaluation_counterpart_bundle"],"sha256":self.cfg["evaluation_counterpart_sha256"]},
            "train_sam_cache":self.cfg["sam_cache"]["manifest"],"eval_sam_cache":self.cfg["sam_eval_cache"]["manifest"]}
    def test_registered_corner_and_existing_stage3(self):
        self.assertTrue(validate_config(self.cfg,"train"))
        self.assertTrue(validate_config(self.parent,"train"))
        validate_parent_input_contract(self.parent,self.cfg)
        bad=copy.deepcopy(self.cfg);bad["training_protocol"]=SAM_NATIVE_PROTOCOL
        with self.assertRaises(ValueError):validate_config(bad,"train")
    def test_wrong_flags_ROI_sources_and_cache_rejected(self):
        for changes in ({"corner_enabled":False},{"sam_enabled":False},{"a0_variant":"a0_sam_image"},
            {"roi_source":"GT"},{"roi_expansion":2.25},{"roi_jitter":{}},{"training_protocol":"unregistered"}):
            bad=copy.deepcopy(self.cfg);bad.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_config(bad,"train")
        bad=copy.deepcopy(self.cfg);bad["sam_eval_cache"]["expected_sources"]["b0_model_sha256"]="0"*64
        with self.assertRaises(ValueError):validate_config(bad,"train")
    def test_SAM_parent_no_GT_fallback_or_input_drift(self):
        bad=copy.deepcopy(self.cfg);bad.pop("training_protocol");bad.update(roi_source="GT",roi_expansion=2.25)
        with self.assertRaises(ValueError):validate_parent_input_contract(self.parent,bad)
        bad=copy.deepcopy(self.cfg);bad["prediction_roi_manifest"]["rows"]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"identities"):validate_parent_input_contract(self.parent,bad)
    def test_legacy_label_and_native_ROI_actually_forwarded(self):
        legacy=_corner_input_contract({"roi_expansion":2.25})
        self.assertEqual(legacy["schema"],"action4.corner_legacy_live_B0_roi225.v1")
        for bad in ({"roi_expansion":1.5},{"roi_source":"frozen_b0_prediction"}):
            with self.assertRaises(ValueError):_corner_input_contract(bad)
        p=SimpleNamespace(acfg=self.cfg,_corner_input_contract=_corner_input_contract(self.cfg),roi_expansion=1.5)
        with patch("eval.action4_corner_prediction.prepare_pipeline_sam",return_value="packet") as prepare:
            self.assertEqual(_prepare_corner_sam(p,"rgb","K",[],"context"),"packet")
            self.assertEqual(prepare.call_args.kwargs,{"roi_expansion":1.5})
        p.roi_expansion=2.25
        with self.assertRaises(ValueError):_prepare_corner_sam(p,"rgb","K",[],"context")
    def replay_fixture(self):
        base=R/"outputs/action4_stage4_native_prediction_readiness_v001/tests"/uuid.uuid4().hex;base.mkdir(parents=True)
        rows=[{"sample_id":"fixture","instances":[{"grid_index":7,"score":.8,"bbox_full_xyxy":[0.,0.,2.,2.]}]}]
        path=base/"predictions.json";path.write_text(json.dumps(rows))
        ref={"path":str(path.relative_to(R)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        cfg=copy.deepcopy(self.cfg);cfg["frozen_eval_b0_predictions"]=ref
        rgb=np.arange(18,dtype=np.uint8).reshape(2,3,3);frame=ImageFrame("fixture",rgb,"native","a"*64,"native_original",array_digest(rgb))
        identity=NativeMaskIdentity("fixture","b0:fixture:grid:7","native")
        context=SamInferenceContext(frame,{native_key(identity):None},cfg["sam_cache"]["expected_sources"])
        p=SimpleNamespace(acfg=cfg,_corner_input_contract=_corner_input_contract(cfg),roi_expansion=1.5,
            _corner_prediction_source_binding=ref,b0_scoring_policy=cfg["b0_scoring_policy"],
            corner_cache=SimpleNamespace(images={"fixture":{}},sources=cfg["sam_cache"]["expected_sources"]),
            _detect_corner=Mock(side_effect=AssertionError("B0 must not run")))
        from eval.action4_corner_prediction import _FrozenB0Source
        p._corner_frozen_source=_FrozenB0Source.load(ref,p.corner_cache.images)
        return p,context,rgb,path
    def test_frozen_replay_does_not_call_B0(self):
        p,c,rgb,_=self.replay_fixture()
        rows=_corner_candidates(p,rgb,np.eye(3),c)
        self.assertEqual(rows[0]["grid_index"],7);p._detect_corner.assert_not_called()
    def test_native1_5_prepares_actual_RGB_affine_and_K(self):
        import torch
        from data.action4_dataset import prepare_rois,virtual_roi
        p,c,rgb,_=self.replay_fixture()
        rows=_corner_candidates(p,rgb,np.eye(3),c)
        rows[0]["prediction_id"]="b0:fixture:grid:7"
        inputs=_prepare_corner_sam(p,rgb,np.eye(3),rows,c)
        reference,matrices=prepare_rois([rgb],[rows[0]["bbox_full_xyxy"]],[0],roi_size=448,expansion=1.5,device="cpu")
        self.assertTrue(torch.equal(inputs.rgb,reference))
        np.testing.assert_array_equal(inputs.A448[0],matrices[0])
        np.testing.assert_array_equal(inputs.K112[0],inputs.A112[0]@np.eye(3))
        np.testing.assert_array_equal(inputs.source_dependencies[0]["roi_bounds_native"],virtual_roi(rows[0]["bbox_full_xyxy"],1.5))
        p._detect_corner.assert_not_called()

    def test_replay_wrong_RGB_ID_source_and_operating_domain_rejected(self):
        p,c,rgb,path=self.replay_fixture()
        changed=rgb.copy();changed[0,0,0]^=1
        with self.assertRaisesRegex(ValueError,"RGB"): _corner_candidates(p,changed,np.eye(3),c)
        foreign=SamInferenceContext(ImageFrame("other",rgb,"native","a"*64,"native_original",array_digest(rgb)),c.predictions,c.expected_sources)
        with self.assertRaises(ValueError):_corner_candidates(p,rgb,np.eye(3),foreign)
        missing=SamInferenceContext(c.frame,{},c.expected_sources)
        with self.assertRaisesRegex(ValueError,"domain"):_corner_candidates(p,rgb,np.eye(3),missing)
        path.write_text(path.read_text()+" ")
        with self.assertRaises(ValueError):_corner_candidates(p,rgb,np.eye(3),c)
    def test_source_parsed_once_returns_copies_and_internal_table_is_immutable(self):
        from eval.action4_corner_prediction import _FrozenB0Source
        import data.action4_sam_native_export as exporter
        p,c,rgb,_=self.replay_fixture()
        with patch.object(exporter,"_v2_checked",wraps=exporter._v2_checked) as read:
            proof=_FrozenB0Source.load(p._corner_prediction_source_binding,p.corner_cache.images)
            p._corner_frozen_source=proof
            a=_corner_candidates(p,rgb,np.eye(3),c)
            a[0]["bbox_full_xyxy"][0]=999
            a[0]["score"]=0.
            b=_corner_candidates(p,rgb,np.eye(3),c)
            self.assertEqual(read.call_count,1)
        self.assertEqual(b[0]["bbox_full_xyxy"][0],0.)
        self.assertEqual(b[0]["score"],.8)
        with self.assertRaises(TypeError):proof.rows["fixture"]="[]"
        with self.assertRaises(TypeError):proof.binding["sha256"]="0"*64
        self.assertIsInstance(proof.rows["fixture"],str)
    def test_source_single_byte_change_rejected_even_with_restored_mtime(self):
        import os
        p,c,rgb,path=self.replay_fixture();st=path.stat()
        raw=bytearray(path.read_bytes());idx=raw.index(ord("8"));raw[idx]=ord("7")
        path.write_bytes(raw);os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,"identity changed"):
            _corner_candidates(p,rgb,np.eye(3),c)
    def test_source_replacement_between_stat_and_sha_memo_rejected(self):
        import data.action4_sam_native_export as exporter
        p,c,rgb,path=self.replay_fixture();real=exporter._v2_sha
        replacement=path.with_name("replacement.json");replacement.write_bytes(path.read_bytes())
        def swap(arg):
            value=real(arg);replacement.replace(path);return value
        with patch.object(exporter,"_v2_sha",side_effect=swap):
            with self.assertRaisesRegex(ValueError,"identity changed"):_corner_candidates(p,rgb,np.eye(3),c)
    def test_fabricated_or_unsealed_proof_rejected(self):
        from dataclasses import replace
        p,c,rgb,_=self.replay_fixture();genuine=p._corner_frozen_source
        for forged in (SimpleNamespace(current_rows=Mock(return_value=[])),replace(genuine)):
            p._corner_frozen_source=forged
            with self.assertRaisesRegex(ValueError,"proof|authenticated"):_corner_candidates(p,rgb,np.eye(3),c)

    def test_inherited_process_proof_rejected(self):
        p,c,rgb,_=self.replay_fixture()
        with patch("eval.action4_corner_prediction.os.getpid",return_value=p._corner_frozen_source.pid+1):
            with self.assertRaisesRegex(ValueError,"current process"):_corner_candidates(p,rgb,np.eye(3),c)

    def test_missing_refined_sidecar_never_uses_old_or_center_quality(self):
        from eval.action4_corner_prediction import _corner_full_box_quality
        p=SimpleNamespace(_corner_input_contract=_corner_input_contract(self.cfg),_corner_box_gate={"threshold":0.})
        passed,reason,scope=_corner_full_box_quality(p,{"score":1.,"center_score":1.,"old_quality_score":1.})
        self.assertFalse(passed);self.assertEqual(reason,"frozen_refined_quality_sidecar_not_bound")
        self.assertEqual(scope,"unadmitted")
        p._corner_input_contract=_corner_input_contract({})
        with self.assertRaisesRegex(ValueError,"refined quality"):
            _corner_full_box_quality(p,{"score":1.})
        from eval.action4_b0_scoring import REFINED_QUALITY_PROTOCOL
        self.assertTrue(_corner_full_box_quality(p,{"refined_quality_protocol":REFINED_QUALITY_PROTOCOL,"refined_quality_score":.9})[0])
    def test_legacy_live_route_preserved(self):
        p=SimpleNamespace(acfg={},_corner_input_contract=_corner_input_contract({}),roi_expansion=2.25,
            _detect_corner=Mock(return_value=[{"legacy":True}]))
        self.assertEqual(_corner_candidates(p,"rgb",None,None),[{"legacy":True}])
        p._detect_corner.assert_called_once_with("rgb")

    def test_new_protocol_never_opens_scheduled_corner_gate(self):
        from model.training.action4_stage2_core import validate_run_gate
        cfg=copy.deepcopy(self.cfg);cfg.pop("a0_selection_contract",None)
        with self.assertRaisesRegex(ValueError,"Corner candidate"):
            validate_run_gate(cfg,{"launch_allowed":True,"mode":"integration"})

if __name__=="__main__":unittest.main()
