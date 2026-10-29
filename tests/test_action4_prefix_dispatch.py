import unittest
from eval.action4_prefix_dispatch import validate_prefix
class PrefixOnly(unittest.TestCase):
 def cfg(self):return dict(schema="action4.sam.b0_video_sequence.v1",discovery_seed_only=True,discovery_decode_prefix=True,seed_mode="box_text",sequence={"direction":"both"})
 def test_prefix_only(self):validate_prefix(self.cfg())
 def test_forward_or_full_clip_refused(self):
  for key,value in [("discovery_seed_only",False),("discovery_decode_prefix",False),("seed_mode","instance_points"),("sequence",{"direction":"forward"})]:
   c=self.cfg();c[key]=value
   with self.assertRaises(ValueError):validate_prefix(c)


class ForwardOnly(unittest.TestCase):
 def test_explicit_completed_prefix_required(self):
  from eval.action4_prefix_dispatch import validate_forward
  c=dict(schema="action4.sam.b0_video_sequence.v1",discovery_seed_only=False,discovery_decode_prefix=False,
      seed_mode="instance_points",instance_output_bridge=True,temporal_mode="causal_forward_B0_assist",
      sequence={"direction":"forward"},instance_seed_sources={"track00":{"config":{}}})
  m={"preparation_provenance":{"kind":"completed_prefix_discovery_immutable_inheritance"}}
  validate_forward(c,m)
  for k,v in [("discovery_seed_only",True),("temporal_mode","causal_forward_A0_face_hypothesis"),("instance_seed_sources",{})]:
   bad=dict(c);bad[k]=v
   with self.assertRaises(ValueError):validate_forward(bad,m)
  with self.assertRaises(ValueError):validate_forward(c,{})
