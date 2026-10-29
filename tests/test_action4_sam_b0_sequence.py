"""Seed and sequence driver tests with explicit synthetic native responses."""
import copy,json,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from model.action4_sam_session_contracts import RegionSpec
from eval.action4_sam_b0_sequence import select_seeds,native_tracks
R=Path("/data/zy/vln/a-test")
class SequenceTests(unittest.TestCase):
 def setUp(self):
  self.region=RegionSpec("fixture",tuple(range(820,824)),tuple(i*400 for i in range(820,824)),(1,11988),"manifest","native",10,10)
  self.policy={"confirmations":3,"score_min":.9,"quality_min":.8,"max_sessions":2}
  self.source={"model_sha256":"b"*64,"source_sha256":"c"*64}
  self.rows=[]
  for i in range(4):
   ps=[]
   for k,box in enumerate([[1.,1.,4.,4.],[6.,6.,9.,9.]]):
    ps.append({"grid_index":k,"bbox_full_xyxy":box,"score":.99-k*.02,"refined_quality_score":.9,"appearance_available":True,
        "appearance_128":np.eye(128)[k].tolist()})
   self.rows.append({"frame":{"global_frame_id":820+i,"local_index":i,"pts":(820+i)*400,"sha256":"a"*64},
                     "pool":{"association_candidates":ps}})
 def test_two_stable_predicted_seeds_without_forcing_pair(self):
  result=select_seeds(self.rows,self.region,self.source,self.policy)
  self.assertEqual(len(result["seeds"]),2)
  one=copy.deepcopy(self.rows)
  for row in one:row["pool"]["association_candidates"]=row["pool"]["association_candidates"][:1]
  self.assertEqual(len(select_seeds(one,self.region,self.source,self.policy)["seeds"]),1)
  for row in one:row["pool"]["association_candidates"][0]["score"]=.5
  self.assertEqual(select_seeds(one,self.region,self.source,self.policy)["seeds"],[])
 def test_unstable_candidate_not_confirmed(self):
  rows=copy.deepcopy(self.rows);rows[1]["pool"]["association_candidates"][1]["bbox_full_xyxy"]=[0.,6.,3.,9.]
  result=select_seeds(rows,self.region,self.source,self.policy)
  self.assertEqual(len(result["seeds"]),1)
 def test_five_frame_confirmation_and_bounds(self):
  rows=copy.deepcopy(self.rows)
  for i in (4,5):
   row=copy.deepcopy(rows[-1]);row["frame"].update(global_frame_id=820+i,local_index=i,pts=(820+i)*400);rows.append(row)
  region=RegionSpec("fixture",tuple(range(820,826)),tuple(i*400 for i in range(820,826)),(1,11988),"manifest","native",10,10)
  result=select_seeds(rows,region,self.source,dict(self.policy,confirmations=5))
  self.assertEqual(len(result["seeds"]),2)
  self.assertEqual(result["seeds"][0]["local_index"],4)
  self.assertEqual(len(result["seeds"][0]["confirmation_scores"]),5)
  with self.assertRaises(ValueError):select_seeds(rows,region,self.source,dict(self.policy,confirmations=7))
 def test_adjacent_duplicate_seed_is_not_second_instance(self):
  rows=copy.deepcopy(self.rows)
  for row in rows:
   row["pool"]["association_candidates"][1]["grid_index"]=10
  duplicate=copy.deepcopy(rows[1]["pool"]["association_candidates"][0])
  duplicate["grid_index"]=1;duplicate["score"]=.98
  rows[1]["pool"]["association_candidates"].append(duplicate)
  result=select_seeds(rows,self.region,self.source,self.policy)
  self.assertEqual(len(result["seeds"]),2)
  self.assertEqual(len(result["duplicate_seed_hypotheses"]),1)
  self.assertEqual(len(rows[1]["pool"]["association_candidates"]),3)
 def test_overlap_alone_does_not_merge_distinct_appearance(self):
  from eval.action4_sam_b0_sequence import deduplicate_seed_candidates
  a,b=copy.deepcopy(self.rows[0]["pool"]["association_candidates"])
  b["bbox_full_xyxy"]=a["bbox_full_xyxy"]
  kept,groups=deduplicate_seed_candidates([a,b])
  self.assertEqual(len(kept),2);self.assertEqual(groups,[])
 def execute_fixture(self,fail=False,seed_only=False):
  base=R/"outputs/action4_clip_replay_preparation_v001";base.mkdir(exist_ok=True)
  temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temp.cleanup);out=Path(temp.name)
  frames=out/"frames";frames.mkdir()
  cfg={"output":str(out.relative_to(R)),"region":{"path":"fixture","sha256":"d"*64},"b0":{"sha256":"b"*64},
       "seed_policy":self.policy,"resources":{"gpu_uuid":"GPU-fixture"},"native":{"text":"camera"},
       "binding":{"mask_in_box_min":.8,"unique_margin":.15,"multi_target_support_min":.7}}
  cfg["discovery_seed_only"]=seed_only
  p=out/"pools.jsonl";p.write_text("".join(json.dumps(x)+"\n" for x in self.rows))
  (out/"B0_COMPLETE.json").write_text(json.dumps({"source_identity":"c"*64,"b0":cfg["b0"],"region":cfg["region"],
       "frames":4,"pools":{"path":str(p.relative_to(R)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}}))
  seed={"track_id":"track","candidate_id":"candidate","local_index":2,"global_frame_id":822,
        "box_xywh01":[.1,.1,.3,.3],"bbox_native":[1.,1.,4.,4.],"evidence_ids":["frozen-B0-fixture"]}
  owner=self;owner.closed=0
  def raw(index):
   m=np.zeros((1,10,10),bool);m[:,1:4,1:4]=True
   return {"frame_index":index,"outputs":{"out_obj_ids":np.array([7]),"out_probs":np.array([.9],np.float32),
           "out_boxes_xywh":np.array([[.1,.1,.3,.3]],np.float32),"out_binary_masks":m}}
  class Fake:
   def __init__(self,c):pass
   def call(self,request):
    if request["type"]=="start_session":return {"session_id":"session"}
    if request["type"]=="add_prompt":return raw(2)
    owner.closed+=1;return {"is_success":True}
   def propagate(self,request):
    if seed_only:raise AssertionError("Seed-only mode must never propagate")
    yield raw(2)
    if fail:raise RuntimeError("injected native failure")
    for i in [3,1,0]:yield raw(i)
   def assert_idle(self):pass
  doc={"frames":[x["frame"] for x in self.rows],"staged_frames":str(frames.relative_to(R))}
  with patch("eval.action4_sam_b0_sequence.context",return_value=({"identity":"c"*64},doc,self.region)), patch("eval.action4_sam_b0_sequence.bound",return_value=doc),        patch("eval.action4_sam_b0_sequence.select_seeds",return_value={"seeds":[seed],"identity_scope":"engineering fixture"}),        patch("eval.action4_sam_b0_sequence.SequenceBackend",Fake):
   if fail:
    with self.assertRaisesRegex(RuntimeError,"injected native"):native_tracks(cfg)
    self.assertFalse((out/"NATIVE_COMPLETE.json").exists());self.assertTrue((out/"track/FAILED.json").exists())
   else:
    native_tracks(cfg);result=json.loads((out/"NATIVE_COMPLETE.json").read_text())
    if seed_only:
     self.assertEqual(result["status"],"DISCOVERY_SEEDS_ONLY_COMPLETE_NOT_TRACKING")
     self.assertFalse(result["tracks"][0]["propagation_executed"])
    else:
     self.assertEqual(result["tracks"][0]["frames"],4);self.assertEqual(result["tracks"][0]["available"],4)
    self.assertFalse(result["accuracy_accepted"]);self.assertFalse(result["association_to_current_B0_complete"])
  self.assertEqual(owner.closed,1)
 def test_discovery_only_never_propagates(self):self.execute_fixture(seed_only=True)
 def test_whole_region_response_and_release(self):self.execute_fixture()
 def test_native_failure_releases_session_and_never_completes(self):self.execute_fixture(True)

 def test_forward_responsibility_preserves_prefix_confirmation_and_native_indices(self):
  from eval.action4_sam_b0_sequence import forward_responsibility,forward_response,sequence_direction
  info=select_seeds(self.rows,self.region,self.source,self.policy)["seeds"][0]
  tail,proof=forward_responsibility(self.region,info)
  self.assertEqual(tail.frame_ids,(822,823))
  self.assertEqual(proof["mapping"][0],{"native_local_index":2,"responsibility_local_index":0,"global_frame_id":822,"pts":328800})
  raw={"frame_index":2,"outputs":None}
  self.assertEqual(forward_response(raw,self.region,tail,2,2)["frame_index"],0)
  self.assertEqual(raw["frame_index"],2)
  with self.assertRaises(ValueError):forward_response(raw,self.region,tail,2,3)
  bad=copy.deepcopy(info);bad["evidence_ids"][-1]="frame823:grid:0"
  with self.assertRaises(ValueError):forward_responsibility(self.region,bad)
  with self.assertRaises(ValueError):sequence_direction({"sequence":{"direction":"forward"}})

if __name__=="__main__":unittest.main(verbosity=2)
