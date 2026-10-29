"""CPU mock backend only; never an actual native pilot outcome."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from model.action4_sam_face_contract import FaceSeedBinding
from model.action4_sam_session_contracts import RegionSpec
from eval.action4_face_native_pilot import run_sessions,ROOT
class PilotTests(unittest.TestCase):
 def fixture(self,mode="good"):
  temp=tempfile.TemporaryDirectory(dir=ROOT/"outputs");self.addCleanup(temp.cleanup)
  out=Path(temp.name);region=RegionSpec("short",(20,21),(200,210),(1,100),"a"*64,"native",20,20)
  ref={"path":"fixture.json","sha256":"b"*64}
  seed=FaceSeedBinding("body",0,0,0,region.identity,0,20,200,(1,100),"native","c"*64,"d"*64,
   (1,0,0,0,1,0,0,0,1),dict(ref),dict(ref),dict(ref),(.1,.1,.8,.8))
  body=np.zeros((20,20),bool);body[2:18,2:18]=True
  mask=np.zeros_like(body);mask[3:9,3:9]=True
  jobs=[dict(seed=seed,points=[[.25,.25],[.35,.35],[.7,.7],[.8,.8]],labels=[1,1,0,0],
             body=body,neighbors=[],other_bodies=[],hypothesis_id="fixture-hypothesis")]
  thresholds=dict(min_positive_coverage=1,max_negative_coverage=0,min_body_containment=.9,
    max_whole_body_iou=.85,max_neighbor_coverage=.1,max_other_body_fraction=.05,max_duplicate_iou=.9)
  cfg={"output":str(out),"resources":{"worker_id":"w","gpu_uuid":"fixture"},"face_pilot":{"thresholds":thresholds}}
  class Fake:
   closed=False
   requests=[]
   def raw(self,i):
    return {"frame_index":i,"outputs":{"out_obj_ids":np.array([0]),"out_probs":np.array([.9]),
     "out_boxes_xywh":np.array([[.1,.1,.8,.8]]),"out_binary_masks":np.array([body if mode=="wholebody" else mask])}}
   def call(self,r):
    self.requests.append(r)
    if r["type"]=="start_session":return {"session_id":"sid"}
    if r["type"]=="add_prompt":return self.raw(0)
    self.closed=True;return {"is_success":True}
   def propagate_forward(self,r):
    if r["propagation_direction"]!="forward":raise AssertionError
    yield self.raw(1 if mode=="future" else 0)
    if mode!="missing":yield self.raw(1)
   def assert_idle(self):
    if not self.closed:raise AssertionError
  return out,cfg,region,jobs,Fake()
 def run_one(self,mode):
  out,cfg,region,jobs,backend=self.fixture(mode)
  value=run_sessions(cfg,{"identity":"source"},{"staged_frames":str(out)},region,jobs,backend)
  return out,value,backend
 def test_public_entry_refuses_long_region_before_gpu_guard(self):
  from eval.action4_face_native_pilot import run
  cfg={"schema":"action4.sam.b0_video_sequence.v1","seed_mode":"instance_points",
       "instance_output_bridge":True,"temporal_mode":"causal_forward_A0_face_hypothesis",
       "sequence_consumer":"A0_face_native_pilot","face_pilot":{},
       "sequence":{"direction":"forward","frame_count":16}}
  with self.assertRaisesRegex(ValueError,"2..15"):run(cfg)
 def test_face_forward_role_is_opt_in_and_B0_runner_refuses(self):
  from eval.action4_sam_b0_sequence import sequence_direction
  cfg={"sequence":{"direction":"forward"},"temporal_mode":"causal_forward_A0_face_hypothesis",
       "sequence_consumer":"A0_face_native_pilot","seed_mode":"instance_points","face_pilot":{}}
  self.assertEqual(sequence_direction(cfg,allow_face=True),"forward")
  with self.assertRaises(ValueError):sequence_direction(cfg)
  from eval.action4_sam_b0_sequence import context
  with self.assertRaises(ValueError):context(cfg)
  bad=dict(cfg,temporal_mode="causal_forward_B0_assist")
  with self.assertRaises(ValueError):sequence_direction(bad,allow_face=True)
  bad=dict(cfg);bad.pop("sequence_consumer")
  with self.assertRaises(ValueError):sequence_direction(bad,allow_face=True)
  self.assertEqual(sequence_direction({"sequence":{"direction":"forward"},"temporal_mode":"causal_forward_B0_assist"}),"forward")
  self.assertEqual(sequence_direction({"sequence":{"direction":"both"}}),"both")
 def test_real_proposal_schema_disk_inputs_load(self):
  from tests.test_action4_face_prompt_proposals import FacePromptTests
  from data.action4_surface_snapshot import make_snapshot,save_snapshot
  from eval.action4_face_native_pilot import load_jobs
  fixture=FacePromptTests(methodName="runTest");fixture.setUp();self.addCleanup(fixture.doCleanups)
  proposal=fixture.run_proposals()
  snapshot=make_snapshot(frame=fixture.frame,prediction=fixture.pred,output=fixture.output,index=0,
      A112=fixture.roi.affine(112),A448=fixture.roi.affine(),K_native=fixture.K,
      K112=fixture.roi.camera(fixture.K,112).astype(np.float32),
      canonical=np.asarray(fixture.contract.canonical8_m,np.float32),models=fixture.models,video_context=fixture.context)
  folder=fixture.folder
  source=folder/"pilot_surface.npz";save_snapshot(source,snapshot)
  body=folder/"pilot_body.npz"
  with body.open("xb") as stream:np.savez_compressed(stream,mask=fixture.mask)
  proposal_ref=fixture.owner.write(folder/"pilot_proposal.json",proposal)
  chosen=[h["hypothesis_identity"] for h in proposal["hypotheses"] if h["status"]=="prompt_hypothesis"]
  cfg={"face_pilot":{"proposal_ref":proposal_ref,"surface_snapshot_ref":fixture.owner.ref(source),
       "body_mask_ref":fixture.owner.ref(body),"expected_models":fixture.models,
       "hypothesis_ids":chosen,"face_generation":0}}
  first=fixture.replay.doc["frames"][3]
  region=RegionSpec("pilot",(first["global_frame_id"],first["global_frame_id"]+1),
       (first["pts"],first["pts"]+400),fixture.replay.region.time_base,"a"*64,fixture.frame.pixel_space_id,40,40)
  jobs=load_jobs(cfg,{"frames":[first]},region)
  self.assertEqual(len(jobs),2)
  self.assertTrue(all(j["seed"].source=="A0_face_hypothesis" for j in jobs))
  cfg["face_pilot"]["expected_models"]={}
  with self.assertRaisesRegex(ValueError,"selected model pair"):load_jobs(cfg,{"frames":[first]},region)
 def test_seed_and_forward_real_adapters(self):
  out,r,b=self.run_one("good")
  self.assertFalse(r["face_tracking_accepted"]);self.assertFalse(r["forward_quality_checked"])
  self.assertEqual(r["results"][0]["status"],"native_identity_complete_not_face_accuracy")
  self.assertTrue((out/"face_00/RAW_RECEIPT.json").is_file())
  self.assertTrue((out/"face_00/response_0001/RAW_RECEIPT.json").is_file())
  self.assertTrue((out/"face_00/CLOSE.json").is_file())
  self.assertEqual(b.requests[1]["obj_id"],0)
 def test_wholebody_unknown_retained_no_forward(self):
  out,r,b=self.run_one("wholebody")
  self.assertEqual(r["results"][0]["status"],"seed_unknown")
  self.assertFalse((out/"face_00/SNAPSHOT.json").exists())
  self.assertTrue(b.closed)
 def test_future_and_missing_preserve_raw_failure_no_complete(self):
  for mode in ("future","missing"):
   out,cfg,region,jobs,b=self.fixture(mode)
   with self.assertRaises(ValueError):
    run_sessions(cfg,{"identity":"source"},{"staged_frames":str(out)},region,jobs,b)
   self.assertTrue((out/"face_00/FAILED.json").is_file())
   self.assertTrue((out/"face_00/response_0000/RAW_RECEIPT.json").is_file())
   self.assertFalse((out/"FACE_PILOT_COMPLETE.json").exists());self.assertTrue(b.closed)
if __name__=="__main__":unittest.main()
