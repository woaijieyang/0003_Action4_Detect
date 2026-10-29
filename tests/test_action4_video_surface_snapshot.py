import copy,unittest
import numpy as np
import torch
from tests.test_action4_surface_snapshot import SurfaceSnapshotTests as Base
from data.action4_video_roi_context import VideoROIContext,VideoSAMSupportReceipt,VideoSAMSupportAdmission,rgb_digest
from data.action4_sam_roi_inputs import array_digest
from eval import action4_surface_snapshot as ss
class VideoSurface(Base):
 def context(self,supported=False):
  rgb=torch.from_numpy(np.moveaxis(self.rgb,-1,0).copy());box=[2.,3.,28.,29.]
  t={'track_id':'left','status':'observed','candidate_id':'b0:frame9:grid:7','bbox_for_roi':box,'raw_detection':{'grid_index':7,'candidate_id':'b0:frame9:grid:7','score':.9,'bbox_full_xyxy':box}}
  row={'time_base':[1,11988],'frame':{'global_frame_id':9,'pts':3600},'session_id':'sam-session','generation':0,'fused':{'mode':'video_inference','frame_id':9,'tracks':[t]}}
  admission=None
  if supported:
   t.update(status='sam_supported',candidate_id=None,raw_detection=None,sam_current_support=True,support_rejection=None)
   mask=np.zeros(self.rgb.shape[:2],bool);mask[3:29,2:28]=True
   r=VideoSAMSupportReceipt(9,3600,(1,11988),rgb_digest(rgb),'native.fixture','left',0,'sam-session',0,0,'b'*64,'c'*64,'d'*64,array_digest(mask),tuple(box))
   admission=VideoSAMSupportAdmission('native.fixture',(1,11988),'c'*64,'d'*64,(r,),((r,mask),))
  models=ss.model_identities(self.p._surface_model_bindings)
  return rgb,VideoROIContext.from_replay(row,rgb,self.K,b0_sha256=models['B0']['sha256'],replay_identity='b'*64,sam_support=admission)
 def test_observed_video_keeps_track_and_real_B0_grid(self):
  rgb,c=self.context();a=self.p.predict_video(rgb,self.K,self.cad,video_context=c)
  b=self.p.predict_video(rgb,self.K,self.cad,video_context=c,return_surface=True,surface_frame=self.frame)
  self.p._detect.assert_not_called()
  self.assertEqual(a['instances'][0]['corners_px'],b['instances'][0]['corners_px'])
  s=b['surface_snapshots'][0]
  self.assertEqual(s.metadata['schema'],ss.VIDEO_SCHEMA);self.assertEqual(s.metadata['prediction_id'],'video:left:frame:9');self.assertEqual(s.metadata['grid_index'],7)
  ss.validate_snapshot(s,frame=self.frame)
 def test_SAM_maintained_video_does_not_invent_B0_grid(self):
  rgb,c=self.context(True)
  s=self.p.predict_video(rgb,self.K,self.cad,video_context=c,return_surface=True,surface_frame=self.frame)['surface_snapshots'][0]
  self.assertIsNone(s.metadata['grid_index']);self.assertIsNone(s.metadata['video_context']['candidate']['score'])
  bad=copy.deepcopy(s.metadata);bad['grid_index']=7;bad['receipt']=ss._canonical({k:v for k,v in bad.items() if k!='receipt'})
  with self.assertRaises(ValueError):ss.validate_snapshot(ss.SurfaceSnapshot(bad,s.arrays))
 def test_resealed_wrong_session_rejected(self):
  rgb,c=self.context(True)
  s=self.p.predict_video(rgb,self.K,self.cad,video_context=c,return_surface=True,surface_frame=self.frame)['surface_snapshots'][0]
  bad=copy.deepcopy(s.metadata);v=bad['video_context'];v['source_ledger']['native_session_id']='wrong'
  v['identity']=ss._canonical({k:x for k,x in v.items() if k!='identity'});bad['receipt']=ss._canonical({k:x for k,x in bad.items() if k!='receipt'})
  with self.assertRaises(ValueError):ss.validate_snapshot(ss.SurfaceSnapshot(bad,s.arrays))
 def test_fixed_B0_changed_resolved_ROI_changes_actual_pixels_and_K(self):
  rgb,c=self.context();first=self.p.predict_video(rgb,self.K,self.cad,video_context=c)
  raw={'grid_index':7,'candidate_id':'b0:frame9:grid:7','score':.9,'bbox_full_xyxy':[2.,3.,28.,29.]}
  row={'time_base':[1,11988],'frame':{'global_frame_id':9,'pts':3600},'session_id':'sam-session','generation':0,
    'fused':{'mode':'video_inference','frame_id':9,'tracks':[{'track_id':'left','status':'observed','candidate_id':raw['candidate_id'],'raw_detection':raw,'bbox_for_roi':[12.,3.,38.,29.]}]}}
  models=ss.model_identities(self.p._surface_model_bindings)
  other=VideoROIContext.from_replay(row,rgb,self.K,b0_sha256=models['B0']['sha256'],replay_identity='b'*64)
  second=self.p.predict_video(rgb,self.K,self.cad,video_context=other)
  self.p._detect.assert_not_called()
  self.assertEqual(first['instances'][0]['bbox_b0_raw'],second['instances'][0]['bbox_b0_raw'])
  self.assertNotEqual(first['instances'][0]['bbox_for_a0'],second['instances'][0]['bbox_for_a0'])
  self.assertFalse(torch.equal(self.p.base.backbone.call_args_list[-2].args[0],self.p.base.backbone.call_args_list[-1].args[0]))
  self.assertNotEqual(first['video_roi_consumption']['instances'][0]['K112'],second['video_roi_consumption']['instances'][0]['K112'])
  for result in [first,second]:
   self.assertEqual(result['instances'][0]['corners_raw_from_resolved_roi'],result['instances'][0]['corners_final'])
 def test_enhanced_surface_provenance_cannot_change_frame_or_body(self):
  rgb,c=self.context(True)
  old=self.p.predict_video(rgb,self.K,self.cad,video_context=c,return_surface=True,surface_frame=self.frame)['surface_snapshots'][0]
  meta=copy.deepcopy(old.metadata);v=meta['video_context'];p=v['candidate']
  from data.action4_roi_contract import RoiSpec
  meta['sam_input']={'roi_spec':RoiSpec.from_full_box(old.arrays['bbox_native'],2.25).record(),'protocol':'action4.video_sam_features.v1','sources':{'sam_model_sha256':'a'*64,'b0_model_sha256':v['B0_sha256'],'replay_identity':v['replay_identity']},
   'current_native_receipt':{'pts':v['source_ledger']['pts'],'time_base':v['source_ledger']['time_base'],'frame_id':v['frame_id'],'RGB_sha256':meta['source_sha256'],'pixel_space_id':meta['pixel_space_id'],'mask_sha256':{p['track_id']:'b'*64}},
   'frame_id':v['frame_id'],'track_id':p['track_id'],'roi_decision_id':p['roi_decision_id'],'bbox_for_a0':p['bbox_for_a0'],
   'still_image_cache_used':False,'availability_reason':'current_video_native_mask'}
  def seal(m):
   v=m['video_context'];v['identity']=ss._canonical({k:x for k,x in v.items() if k!='identity'});m['receipt']=ss._canonical({k:x for k,x in m.items() if k!='receipt'})
   return ss.SurfaceSnapshot(m,old.arrays)
  v['SAM_mask_features_enabled']=True;ss.validate_snapshot(seal(meta))
  for key,value in [('track_id','other'),('frame_id',10),('bbox_for_a0',[0,0,1,1]),('still_image_cache_used',True)]:
   bad=copy.deepcopy(meta);bad['sam_input'][key]=value
   with self.assertRaises(ValueError):ss.validate_snapshot(seal(bad))
  bad=copy.deepcopy(meta);bad['video_context']['SAM_mask_features_enabled']=False
  with self.assertRaises(ValueError):ss.validate_snapshot(seal(bad))
del Base
