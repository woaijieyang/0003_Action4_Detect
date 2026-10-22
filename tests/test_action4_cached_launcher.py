"""Cached-launcher control-plane fixtures; never execute Blender/GPU or accept fixture data."""
from pathlib import Path
import copy,json,tempfile,unittest
from model.training import action4_dispatch as d
ROOT=Path(__file__).resolve().parents[1]
class CachedLauncherTests(unittest.TestCase):
 def setUp(self):
  base=ROOT/'outputs/action4_cached_launcher_control_fixtures';base.mkdir(exist_ok=True);self.case=Path(tempfile.mkdtemp(prefix='case_',dir=base));asset=self.case/'asset.txt';asset.write_text('CONTROL_PLANE_FIXTURE_NOT_A_MODEL');config=self.case/'config.json';config.write_text('{"fixture":true}')
  self.plan=self.case/'plan.json';self.plan.write_text(json.dumps({'recipe':{'resolution_wh':[3248,2464],'render':{'samples':64,'mask_samples':1}},'asset_identities':{str(asset.relative_to(ROOT)):d.sha256(asset)}}))
  data='data/action4_synth/control_fixture_never_rendered/'+self.case.name
  from data.action4_synthesis import _run_output_dir
  self.data=data;self.runtime=str(_run_output_dir(data));self.q=d.Dispatcher(ROOT)
  self.spec={'job_id':self.case.name,'kind':'render','cwd':str(ROOT),'argv':['.venv/bin/python','scripts/action4_render_cached.py','render-shard','--plan',str(self.plan.relative_to(ROOT)),'--output',data,'--blender',str(asset.relative_to(ROOT)),'--threads','2','--gpu-uuid',d.ALLOWED_GPU_UUIDS[0]],'output_dir':str(self.case/'attempt'),'result_dirs':[str(ROOT/data),self.runtime],'allowed_gpu_uuids':[d.ALLOWED_GPU_UUIDS[0]],'gpu_count':1,'cpu_threads':2,'reservation_mib':8192,'profile_key':'CONTROL_FIXTURE','configs':{str(config.relative_to(ROOT)):d.sha256(config)},'inputs':{str(self.plan.relative_to(ROOT)):d.sha256(self.plan)},'code':{'scripts/action4_render_cached.py':d.sha256(ROOT/'scripts/action4_render_cached.py')}}
 def test_same_render_resource_contract(self):
  x=self.q._validate(self.spec,profile=True);self.assertEqual(x['effective_workload']['render_threads'],2);self.assertEqual(x['effective_workload']['render_recipe']['resolution_wh'],[3248,2464]);self.assertEqual(x['cpu_threads'],2)
 def test_explicit_threads_required(self):
  x=copy.deepcopy(self.spec);i=x['argv'].index('--threads');del x['argv'][i:i+2]
  with self.assertRaisesRegex(ValueError,'explicit --threads'):self.q._validate(x,profile=True)
 def test_thread_reservation_cannot_understate(self):
  x=copy.deepcopy(self.spec);x['cpu_threads']=1
  with self.assertRaises(ValueError):self.q._validate(x,profile=True)
 def test_runtime_directory_must_be_registered(self):
  x=copy.deepcopy(self.spec);x['result_dirs']=[str(ROOT/self.data)]
  with self.assertRaisesRegex(ValueError,'runtime output'):self.q._validate(x,profile=True)
 def test_internal_entry_rejected(self):
  x=copy.deepcopy(self.spec);x['argv'][2]='_render-shard'
  with self.assertRaisesRegex(ValueError,'public render-shard'):self.q._validate(x,profile=True)
 def test_fixture_or_old_plan_cannot_enqueue(self):
  state=ROOT/'outputs/action4_dispatch/queue.json';before=state.read_bytes() if state.exists() else None
  with self.assertRaises((ValueError,FileNotFoundError)):self.q.enqueue(self.spec,profile=True)
  self.assertEqual(state.read_bytes() if state.exists() else None,before);self.assertFalse((self.case/'attempt').exists());self.assertFalse((ROOT/self.data).exists())
if __name__=='__main__':unittest.main()
