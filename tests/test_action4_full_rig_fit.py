"""Pure algebra/guard tests. Toy fixtures are not real-scene evidence."""
import math
import numpy as np
import unittest
from data.action4_full_rig_fit import euler_xyz,proper,project,validate_pose,fk_bones,translation,_clip,bounded,snapshot_digest,validate_snapshot

def toy():
 I=np.eye(4);F=I.copy();F[1,3]=1;H=I.copy();H[1,3]=2
 return {"bones":{"torso_root":{"parent":None,"rest_matrix":I.tolist(),"length_m":1},
 "left_forearm":{"parent":"torso_root","rest_matrix":F.tolist(),"length_m":1},
 "left_hand":{"parent":"left_forearm","rest_matrix":H.tolist(),"length_m":1}}}


class FullRigForwardTests(unittest.TestCase):
    def test_blender_xyz_order_known_rotation(self):
     R=euler_xyz([0,0,90])
     assert np.allclose(R@np.array([1,0,0]),[0,1,0],atol=1e-14)
     assert np.allclose(R.T@R,np.eye(3),atol=1e-14)

    def test_source_hierarchy_fk_translation(self):
     P=fk_bones(toy(),{"left_forearm":[0,0,25]})
     assert np.allclose(P["left_hand"][:3,3],[-math.sin(math.radians(25)),1+math.cos(math.radians(25)),0])
     assert np.allclose(P["left_forearm"][:3,3],[0,1,0])

    def test_hand_only_does_not_move_forearm(self):
     a=fk_bones(toy(),{});b=fk_bones(toy(),{"left_hand":[15,10,-8]})
     assert np.array_equal(a["left_forearm"],b["left_forearm"])
     assert not np.allclose(a["left_hand"][:3,:3],b["left_hand"][:3,:3])

    def test_pose_boundary_and_no_extra_controls(self):
     validate_pose({"left_hand":[25,-25,0]})
     for p in [{"left_hand":[25.001,0,0]},{"left_hand":[True,0,0]},{"torso_root":[0,0,0]},{"scale":[1,1,1]},{"left_hand":[float("nan"),0,0]}]:
      with self.assertRaises(ValueError):validate_pose(p)

    def test_no_reflection_scale_or_wrong_homogeneous_row(self):
     for i in range(3):
      T=np.eye(4)
      if i==0:T[0,0]=-1
      elif i==1:T[1,1]=1.1
      else:T[3,0]=1
      with self.assertRaises(ValueError):proper(T)

    def test_behind_camera_not_clamped_to_fake_projection(self):
     p=project([[0,0,-1],[0,0,.001],[0,0,1],[100,0,1]],1000)
     assert p["xy"][:2]==[None,None]
     assert p["xy"][2]==[1623.5,1231.5]
     assert p["inside_native"]==[False,False,True,False]

    def test_same_observer_anchor_relative_constraint(self):
     FL=translation(.2);FR=np.eye(4);FR[0,3]=.8;BL=np.eye(4);BL[:3,:3]=euler_xyz([5,6,7]);BL[2,3]=.5
     E=BL@np.linalg.inv(FL);BR=E@FR
     assert np.allclose(E@FL,BL)
     assert np.allclose(np.linalg.inv(BL)@BR,np.linalg.inv(FL)@FR)

    def test_missing_parent_or_cycle_rejected(self):
     s=toy();s["bones"]["torso_root"]["parent"]="left_hand"
     with self.assertRaises(ValueError):fk_bones(s,{})

    def test_clipping_and_project_boundary(self):
     assert _clip(None,[1,1]) is None
     assert _clip([-20,-20],[-5,-5]) is None
     a,b=_clip([-10,100],[30,100]);assert np.allclose(a,[0,100]) and np.allclose(b,[30,100])
     with self.assertRaises(ValueError):bounded("/data/zy/vln/not_this_project")


    def test_snapshot_payload_digest_detects_edit(self):
     a={"bones":{"one":{"length_m":1}},"status":"diagnostic_only"}
     digest=snapshot_digest(a);a["invariants_sha256"]=digest
     assert snapshot_digest(a)==digest
     a["bones"]["one"]["length_m"]=2
     assert snapshot_digest(a)!=digest

    def test_snapshot_stale_digest_rejected_before_source_reads(self):
     value={"schema":"action4.full_rig_forward_snapshot.v1","status":"diagnostic_only","dataset_generation_allowed":False,"invariants_sha256":"bad"}
     with self.assertRaises(ValueError):validate_snapshot(value)

    def test_snapshot_incomplete_bone_coverage_rejected(self):
     value={"schema":"action4.full_rig_forward_snapshot.v1","status":"diagnostic_only","dataset_generation_allowed":False,"bones":{},"sources":{}}
     value["invariants_sha256"]=snapshot_digest(value)
     with self.assertRaises(ValueError):validate_snapshot(value)

if __name__=="__main__":unittest.main()
