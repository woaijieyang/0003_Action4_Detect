import unittest,numpy as np
from eval.action4_corner_video_threshold_replay import restore_route,verify_projection
from data.action4_sam_roi_inputs import array_digest
class TestReplay(unittest.TestCase):
 def test_mask_identity_is_required(self):
  mask=np.ones((3,3),dtype=bool);record={'route':{'kind':'coarse','full_box_quality_passed':False,'support_mask_sha256':array_digest(mask),'support_mask_pixels':9}}
  self.assertIs(restore_route(record,mask)['support_mask_native'],mask)
  with self.assertRaises(ValueError):restore_route(record,None)
  changed=mask.copy();changed[0,0]=False
  with self.assertRaises(ValueError):restore_route(record,changed)
 def test_projection_rejects_changed_points(self):
  cad=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],float);r=np.eye(3).tolist();t=[0,0,5];camera=cad+np.array(t);points=(camera[:,:2]/camera[:,2:]).tolist();p={'final_pose':{'rotation':r,'translation_m':t},'corners_final':points,'corners_px':points,'rotation':r,'translation_m':t,'pose_valid':True}
  verify_projection(p,np.eye(3),cad)
  p['corners_final'][0][0]+=1
  with self.assertRaises(ValueError):verify_projection(p,np.eye(3),cad)
if __name__=='__main__':unittest.main()
