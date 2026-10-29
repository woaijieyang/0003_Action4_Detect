"""Mock native stream contract tests; no GPU/model execution."""
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from eval.action4_sam_sequence_backend import SequenceBackend,SCHEMA
class SequenceTests(unittest.TestCase):
 def setUp(self):
  self.cfg={"schema":SCHEMA,"sequence":{"frame_count":120,"direction":"both","output_prob_thresh":.5,"evict_cached_frame_outputs":False}}
  self.backend=SequenceBackend(self.cfg);self.requests=[];self.closed=False
  owner=self
  class Predictor:
   _all_inference_states={"session":{}}
   def handle_stream_request(self,request):
    owner.requests.append(request)
    try:
     yield {"frame_index":2,"outputs":None}
     yield {"frame_index":0,"outputs":None}
    finally:owner.closed=True
  self.backend.predictor=Predictor()
  self.backend.torch=SimpleNamespace(inference_mode=nullcontext,autocast=lambda **kw:nullcontext(),
       bfloat16="fixture",cuda=SimpleNamespace(synchronize=lambda:None))
  self.a=patch("eval.action4_sam_sequence_backend.validate_live_authority");self.a.start();self.addCleanup(self.a.stop)
  self.b=patch("eval.action4_sam_sequence_backend.validate_source_snapshot");self.b.start();self.addCleanup(self.b.stop)
 def req(self,**kw):return dict({"type":"propagate_in_video","session_id":"session","start_frame_index":2},**kw)
 def test_stream_preserves_raw_order_and_missing_outputs(self):
  rows=list(self.backend.propagate(self.req()))
  self.assertEqual([x["frame_index"] for x in rows],[2,0])
  self.assertIsNone(rows[0]["outputs"]);self.assertTrue(self.closed)
  self.assertFalse(self.requests[0]["evict_cached_frame_outputs"])
  self.assertEqual(self.requests[0]["output_prob_thresh"],.5)
 def test_no_reprompt_or_close_midstream(self):
  stream=self.backend.propagate(self.req());next(stream)
  with self.assertRaisesRegex(RuntimeError,"Close stream"):
   self.backend.call({"type":"close_session","session_id":"session"})
  with self.assertRaisesRegex(RuntimeError,"serial"):next(self.backend.propagate(self.req()))
  stream.close();self.assertIsNone(self.backend._active_stream);self.assertTrue(self.closed)
 def test_illegal_mapping_chunk_or_direction_refused(self):
  for req in [self.req(start_frame_index=820),self.req(max_frame_num_to_track=10),
              self.req(propagation_direction="forward"),self.req(session_id="other")]:
   with self.assertRaises(ValueError):list(self.backend.propagate(req))
  self.assertFalse(self.requests)
 def test_resource_failure_before_native_call(self):
  with patch("eval.action4_sam_sequence_backend.validate_live_authority",side_effect=ValueError("revoked")):
   with self.assertRaisesRegex(ValueError,"revoked"):list(self.backend.propagate(self.req()))
  self.assertFalse(self.requests);self.assertIsNone(self.backend._active_stream)
 def test_exception_clears_busy_state_not_success(self):
  def bad(request):
   yield {"frame_index":0}
   raise RuntimeError("native failure")
  self.backend.predictor.handle_stream_request=bad
  with self.assertRaisesRegex(RuntimeError,"native failure"):list(self.backend.propagate(self.req()))
  self.assertIsNone(self.backend._active_stream)
 def test_forward_requires_opt_in_and_preserves_native_direction(self):
  with self.assertRaises(ValueError):self.backend.propagate_forward(self.req())
  self.backend.config["causal_forward_corrections"]=True
  rows=list(self.backend.propagate_forward(self.req(propagation_direction="forward")))
  self.assertEqual(len(rows),2);self.assertEqual(self.requests[-1]["propagation_direction"],"forward")
 def test_correction_can_run_only_after_stream_close(self):
  stream=self.backend.propagate(self.req());next(stream)
  with self.assertRaises(RuntimeError):self.backend.call({"type":"add_prompt"})
  stream.close();self.assertIsNone(self.backend._active_stream)
 def test_explicit_schema_and_model_required(self):
  with self.assertRaises(ValueError):SequenceBackend(dict(self.cfg,schema="single_image"))
  self.backend.predictor=None
  with self.assertRaisesRegex(RuntimeError,"not loaded"):list(self.backend.propagate(self.req()))

 def test_causal_sequence_explicit_forward_api_and_no_both_fallback(self):
  import copy
  cfg=copy.deepcopy(self.cfg);cfg["sequence"]["direction"]="forward"
  with self.assertRaises(ValueError):SequenceBackend(cfg)
  cfg["temporal_mode"]="causal_forward_B0_assist"
  backend=SequenceBackend(cfg)
  backend.predictor=self.backend.predictor;backend.torch=self.backend.torch
  with self.assertRaises(ValueError):backend.propagate(self.req())
  rows=list(backend.propagate_forward(self.req(propagation_direction="forward")))
  self.assertEqual(len(rows),2)
  self.assertEqual(self.requests[-1]["propagation_direction"],"forward")
  bad=copy.deepcopy(self.cfg);bad["temporal_mode"]="causal_forward_B0_assist"
  with self.assertRaises(ValueError):SequenceBackend(bad)

if __name__=="__main__":unittest.main(verbosity=2)
