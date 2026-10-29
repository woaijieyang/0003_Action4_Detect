"""Trial-selection metadata fixtures; never model/video accuracy proof."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from eval.action4_video_contract import validate_selection
R=Path("/data/zy/vln/a-test")
class TrialTests(unittest.TestCase):
 def setUp(self):
  out=R/"outputs/action4_video_trial_preparation_v001";out.mkdir(exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=out);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  def make(rel,data):
   p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o444)
   return {"path":rel,"sha256":hashlib.sha256(data).hexdigest()}
  self.make=make;self.models={}
  self.cfg={"video_mode":"research_trial","b0_scoring_policy":"center","worker_memory_budget_mib":4096,
            "trial_gpu_uuid":"GPU-fixture","trial_worker_cpu_groups":[[90,91,92,93,94,95]]}
  for role in ("A0","B0"):
   b=make("models/candidates/trial/"+role+".pt",b"METADATA FIXTURE NOT MODEL")
   self.cfg[role.lower()+"_bundle"]=b["path"];self.cfg[role.lower()+"_sha256"]=b["sha256"]
   self.models[role]=dict(b,strict_load=True,selected_step=10,config_id="fixture")
  inputs={}
  for key,rel in (("intrinsics_context","context.json"),("demo_reference","demo.mp4"),
      ("frames_manifest","data/action4_real/cache_v001/frames_manifest.json"),
      ("canonical_asset","data/DJI_Action4_construct/r3_v001/asset_manifest.json")):
   inputs[key]=make(rel,b"metadata-fixture")
   if key in ("intrinsics_context","demo_reference"):self.cfg[key]=rel
  self.receipt={"schema":"action4.research_video_trial.v1","status":"FROZEN_FOR_RESEARCH_VIDEO",
    "authorization_kind":"user_requested_video_trial_during_training","deployment_release":False,"default_models_modified":False,
    "models":self.models,"video_inputs":inputs,"b0_scoring_policy":"center","resources":{"gpu_uuid":self.cfg["trial_gpu_uuid"],
      "worker_cpu_groups":self.cfg["trial_worker_cpu_groups"],"worker_memory_budget_mib":4096}}
  self.publish()
 def publish(self):
  path=self.root/"trial.json"
  if path.exists():path.chmod(0o644)
  self.cfg["trial_receipt"]=self.make("trial.json",json.dumps(self.receipt).encode())
 def test_trial_is_explicit_not_formal(self):
  result=validate_selection(self.cfg,self.root)
  self.assertEqual(result["evaluation_kind"],"research_video_trial")
  bad=copy.deepcopy(self.cfg);bad.pop("video_mode")
  with self.assertRaisesRegex(ValueError,"sealed selected"):validate_selection(bad,self.root)
 def test_immutable_models_and_inputs(self):
  p=self.root/self.cfg["a0_bundle"];p.chmod(0o644)
  with self.assertRaisesRegex(ValueError,"model must be frozen"):validate_selection(self.cfg,self.root)
  p.chmod(0o444);self.cfg["b0_scoring_policy"]="learned"
  with self.assertRaisesRegex(ValueError,"policy changed"):validate_selection(self.cfg,self.root)
 def test_fake_deployment_or_missing_authorization_refused(self):
  for key,value in (("deployment_release",True),("authorization_kind","implicit"),("status","pending")):
   old=self.receipt[key];self.receipt[key]=value;self.publish()
   with self.assertRaises(ValueError):validate_selection(self.cfg,self.root)
   self.receipt[key]=old;self.publish()
 def test_renderer_exact_frame_and_CAD_files(self):
  self.receipt["video_inputs"]["frames_manifest"]=self.make("other.json",b"metadata-fixture");self.publish()
  with self.assertRaisesRegex(ValueError,"actual renderer"):validate_selection(self.cfg,self.root)
 def test_trial_resource_binding(self):
  for key,value in (("trial_gpu_uuid","wrong"),("trial_worker_cpu_groups",[[24]]),("worker_memory_budget_mib",8192)):
   bad=copy.deepcopy(self.cfg);bad[key]=value
   with self.assertRaises(ValueError):validate_selection(bad,self.root)
 def test_no_implicit_tracking_path(self):
  with self.assertRaisesRegex(ValueError,"tracking/session"):validate_selection(dict(self.cfg,sam_enabled=True),self.root)

 def test_completed_comparison_requires_sealed_lineage_membership(self):
  self.receipt['authorization_kind']='user_requested_completed_training_candidate_comparison'
  a=self.models['A0'];b=self.models['B0']
  seal={'schema':'action4.sealed_lineage_training_candidate.v1','role':'A0','strict_load':True,
        'completion':{'formal_training_complete':True,'epochs':80},
        'retained_top_candidates':[{'record':{'step':a['selected_step'],'config_id':a['config_id']},'retained_checkpoint':{'sha256':a['sha256']}}]}
  self.receipt['completed_lineage_seal']=self.make('lineage.json',json.dumps(seal).encode())
  bseal={'schema':'action4.sealed_independent_role.v2','role':'B0','strict_load':True,'selected_step':b['selected_step'],'config_id':b['config_id'],'sha256':b['sha256']}
  self.receipt['fixed_B0_seal']=self.make('b0seal.json',json.dumps(bseal).encode());self.publish()
  self.assertEqual(validate_selection(self.cfg,self.root)['evaluation_kind'],'research_video_trial')
  seal['retained_top_candidates'][0]['record']['step']=999
  p=self.root/'lineage.json';p.chmod(0o644)
  self.receipt['completed_lineage_seal']=self.make('lineage.json',json.dumps(seal).encode());self.publish()
  with self.assertRaises(ValueError):validate_selection(self.cfg,self.root)

if __name__=="__main__":unittest.main(verbosity=2)
