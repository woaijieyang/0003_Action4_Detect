import unittest,copy
import numpy as np
import torch
from data.action4_training_crop_specs import TrainingCropSpec,validate_crop_specs
from data.action4_pose_rgb_batch import prepare_pose_rgb

class CropSpecTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        cad=np.array([[x,y,z] for z in (-.01,.01) for y in (-.02,.02) for x in (-.03,.03)])
        K=np.array([[64.,0,31.5],[0,64.,31.5],[0,0,1]])
        T=np.eye(4);T[2,3]=1
        h=(cad+T[:3,3])@K.T;uv=h[:,:2]/h[:,2:]
        box=np.r_[uv.min(0),uv.max(0)].tolist()
        obj={'instance_id':1,'bbox_full_xyxy':box,'canonical8_geometry':{'object_to_capture_cv_m':T.tolist()},'corners_px':uv.tolist(),'corner_visible':[True]*4+[False]*4,'occlusion_inframe':.5}
        record={'sample_id':'engineering_crop','split':'train','camera':{'K':K.tolist()},'corners_3d_mm':(cad*1000).tolist(),'instances':[obj]}
        self.item={'record':record,'instance':0,'image':torch.arange(3*64*64,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3,64,64)}
    def spec(self,**changes):
        kw=dict(image_id='engineering_crop',instance_id='1',box_xyxy=(12.25,8.5,44.75,41.0),requested_route='prediction_full',route='prediction_full',augmentation_seed=19,policy_sha256='e'*64,reason='engineering_transport_equivalence',prediction_id='b0:engineering_crop:grid:7')
        kw.update(changes);return TrainingCropSpec(**kw)
    def test_same_explicit_box_preserves_rgb_geometry(self):
        spec=self.spec();old=prepare_pose_rgb([self.item],boxes=[list(spec.box_xyxy)])
        new=prepare_pose_rgb([self.item],crop_specs=[spec])
        self.assertTrue(torch.equal(old[0],new[0]))
        for k in old[1]:self.assertTrue(torch.equal(old[1][k],new[1][k]),k)
        for k in ('crop_box','affine_hm','K_hm','corners_hm','visible','diagonal'):self.assertEqual(old[2][0][k],new[2][0][k])
        self.assertEqual(new[2][0]['crop_spec']['prediction_id'],spec.prediction_id)
        self.assertEqual(new[2][0]['crop_mode'],'crop_spec')
    def test_negative_boundaries_are_continuous_and_not_clipped(self):
        spec=self.spec(box_xyxy=(-12.25,-8.5,20.25,24.0))
        rgb,g,meta=prepare_pose_rgb([self.item],crop_specs=[spec])
        self.assertEqual(meta[0]['crop_box'][0],-12.25)
        self.assertTrue(torch.isfinite(rgb).all())
        expected=(-torch.tensor([.485,.456,.406])/torch.tensor([.229,.224,.225]))[None]
        self.assertTrue(torch.equal(rgb[:,:,0,0],expected))
    def test_invalid_identity_and_evaluation_rejected(self):
        with self.assertRaises(ValueError):prepare_pose_rgb([self.item],crop_specs=[self.spec(instance_id='2')])
        item=copy.deepcopy(self.item);item['record']['split']='eval'
        with self.assertRaises(ValueError):prepare_pose_rgb([item],crop_specs=[self.spec()])
    def test_exclusive_selection_and_bad_specs(self):
        for extra in (dict(boxes=[[1,2,3,4]]),dict(seeds=[1]),dict(jitter_config={'center_fraction':.15,'scale_range':[.65,1.35]})):
            with self.assertRaises(ValueError):prepare_pose_rgb([self.item],crop_specs=[self.spec()],**extra)
        for changes in ({'box_xyxy':(1,2,float('nan'),4)},{'box_xyxy':(1,2,1,4)},{'augmentation_seed':True},{'route':'unknown'},{'prediction_id':None}):
            with self.assertRaises(ValueError):self.spec(**changes)
        with self.assertRaises(ValueError):validate_crop_specs([self.item],[])
    def test_fallback_and_box_are_immutable_metadata(self):
        original=[1.,2.,5.,7.]
        spec=self.spec(box_xyxy=original,route='gt_jitter',prediction_id=None,reason='predicted_roi_missing')
        original[0]=100
        self.assertEqual(spec.box_xyxy,(1.,2.,5.,7.))
        _,_,meta=prepare_pose_rgb([self.item],crop_specs=[spec])
        self.assertEqual(meta[0]['crop_spec']['requested_route'],'prediction_full')
        self.assertEqual(meta[0]['crop_spec']['route'],'gt_jitter')

if __name__=='__main__':unittest.main()
