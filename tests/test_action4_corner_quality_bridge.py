import unittest,copy
from eval.action4_corner_quality_bridge import verify_equivalent_capture

class BridgeEquivalenceTests(unittest.TestCase):
    def sample(self):
        base=[dict(sample_id="a",instances=[dict(grid_index=1,score=.9,bbox_full_xyxy=[0,0,10,10])])]
        old=copy.deepcopy(base);old[0]["instances"][0]["refined_quality_score"]=.95
        side=dict(rows=[dict(image=dict(sample_id="a"),qualities=[dict(grid_index=1,refined_quality_score=.95)])])
        return old,side,base
    def test_exact(self):
        self.assertEqual(verify_equivalent_capture(*self.sample()),1)
    def test_different_q_rejected(self):
        a,s,b=self.sample();s["rows"][0]["qualities"][0]["refined_quality_score"]+=1e-12
        with self.assertRaises(ValueError):verify_equivalent_capture(a,s,b)
    def test_different_box_rejected(self):
        a,s,b=self.sample();b[0]["instances"][0]["bbox_full_xyxy"][0]=.001
        with self.assertRaises(ValueError):verify_equivalent_capture(a,s,b)
    def test_missing_or_duplicate_domain(self):
        a,s,b=self.sample()
        with self.assertRaises(ValueError):verify_equivalent_capture(a,s,[])
        with self.assertRaises(ValueError):verify_equivalent_capture(a*2,dict(rows=s["rows"]*2),b*2)

class QualityPreflightTests(unittest.TestCase):
    def config(self):
        return dict(evaluation_counterpart_bundle="model.pt",evaluation_counterpart_sha256="a"*64,
            b0_full_box_quality_receipt={"path":"gate.json","sha256":"b"*64},
            corner_refined_quality_bridge={"path":"bridge.json","sha256":"c"*64})
    def test_bridge_binds_actual_checkpoint_metadata(self):
        from unittest.mock import patch
        from data.action4_sam_native_export import canonical
        from eval.action4_corner_quality_bridge import validate_eval_quality_dependencies
        cfg=self.config();state=dict(role="B0",architecture="online_b0",config={"actual":True},config_id=canonical({"actual":True}),selected_step=17)
        with patch("data.action4_sam_native_export._v2_read",return_value=state),patch("eval.action4_corner_quality_bridge.CornerQualityBridge") as bridge:
            bridge.return_value._gate={"threshold":.5}
            self.assertEqual(validate_eval_quality_dependencies(cfg,{"path":"base","sha256":"d"*64}),{"threshold":.5})
            self.assertEqual(bridge.call_args.kwargs["b0_step"],17)
            self.assertEqual(bridge.call_args.kwargs["b0_config_id"],state["config_id"])
    def test_invalid_model_metadata_never_constructs_bridge(self):
        from unittest.mock import patch
        from eval.action4_corner_quality_bridge import validate_eval_quality_dependencies
        with patch("data.action4_sam_native_export._v2_read",return_value={"role":"A0"}),patch("eval.action4_corner_quality_bridge.CornerQualityBridge") as bridge:
            with self.assertRaises(ValueError):validate_eval_quality_dependencies(self.config(),{})
            bridge.assert_not_called()
    def test_no_bridge_keeps_exact_original_capture_requirement(self):
        from unittest.mock import patch
        from eval.action4_corner_quality_bridge import validate_eval_quality_dependencies
        cfg=self.config();cfg.pop("corner_refined_quality_bridge");base={"path":"base","sha256":"d"*64}
        with patch("eval.action4_corner_quality_bridge.validate_full_box_quality_gate",return_value={}) as validate:
            validate_eval_quality_dependencies(cfg,base)
            self.assertEqual(validate.call_args.kwargs["expected_reference_predictions"],base)

class PredictionAliasTests(unittest.TestCase):
    def test_distinct_paths_require_equal_verified_bytes(self):
        from unittest.mock import patch
        from data.action4_prediction_roi_data import verify_prediction_binding_alias
        a={"path":"producer.json","sha256":"a"*64};b={"path":"consumer.json","sha256":"a"*64}
        with patch("data.action4_sam_native_export._v2_sha",return_value="a"*64):
            r=verify_prediction_binding_alias(a,b)
            self.assertEqual(r["producer"],a);self.assertEqual(r["consumer"],b)
        with patch("data.action4_sam_native_export._v2_sha",side_effect=["a"*64,"b"*64]):
            with self.assertRaises(ValueError):verify_prediction_binding_alias(a,b)
        with self.assertRaises(ValueError):verify_prediction_binding_alias(a,dict(b,sha256="c"*64))
