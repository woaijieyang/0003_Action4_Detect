import copy,unittest
from eval.action4_corner_capacity import validate
class CornerCapacity(unittest.TestCase):
 def config(self,n):
  return dict(capacity_workers=n,mode="integration",evaluation_only_engineering=True,
    training_gpus=[],a0_variant="a0_corner_image",corner_enabled=True,sam_enabled=True,
    expected_eval_images=650,eval_indices=None,eval_loader_workers=0 if n==32 else 1,eval_threads=1,
    eval_gpus=([4]*8+[5]*8+[6]*8+[7]*8 if n==32 else [4]*4+[5]*4+[6]*4+[7]*4 if n==16 else [4]*5+[5]*5+[6]*5+[7]*4),
    process_cpu_affinity=dict({"eval_"+str(i):([24+i] if n==32 else [24+2*i,25+2*i]) for i in range(n)},train=[57] if n==32 else [63]),
    controller_cpu=[56] if n==32 else [62],evaluation_allocator_budget_mib=3072 if n==32 else 4096)
 def test_bounded_topologies(self):
  for n in (16,19,32):self.assertEqual(validate(self.config(n)),n)
 def test_wrong_scope_and_budget(self):
  for key,value in [("mode","full_training"),("training_gpus",[4]),("expected_eval_images",64),
    ("eval_indices",[0]),("evaluation_only_engineering",False),("sam_enabled",False),
    ("evaluation_allocator_budget_mib",8000)]:
   c=self.config(19);c[key]=value
   with self.assertRaises(ValueError):validate(c)
 def test_overlap_or_single_card_overload(self):
  c=self.config(19);c["process_cpu_affinity"]["eval_1"]=[24,25]
  with self.assertRaises(ValueError):validate(c)
  c=self.config(19);c["eval_gpus"]=[4]*19
  with self.assertRaises(ValueError):validate(c)
  c=self.config(19);c["process_cpu_affinity"]["eval_0"]=[24,24]
  with self.assertRaises(ValueError):validate(c)
