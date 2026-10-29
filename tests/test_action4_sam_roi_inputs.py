"""Explicit synthetic fixtures; no SAM execution or formal cache claimed."""
import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
from data.action4_sam_roi_inputs import (ImageFrame,RoiRequest,NativeSamPrediction,SOURCE_KEYS,
    array_digest,identity_key,native_key,NativeMaskIdentity,verify_image_frame,verify_native_prediction,prediction_metadata,cache_key,prepare_sam_roi_inputs,
    prepare_quality_targets,read_prediction_cache,sample_native_nearest)
from data.action4_dataset import prepare_rois
from data.action4_adapter import DataAdapter
from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux,prepare_mask_aux

class SamRoiInputTests(unittest.TestCase):
    def setUp(self):
        self.sources={k:'a'*64 for k in SOURCE_KEYS}
        self.mask=np.zeros((448,448),bool);self.mask[60:180,90:220]=True
        pixels=np.zeros((448,448,3),np.uint8);pixels[:,:,0]=self.mask*255
        self.frame=ImageFrame('image-A',pixels,'native-v1','b'*64,'original',array_digest(pixels))
        self.identity=FeatureIdentity('image-A','left','roi-left','roi448')
        self.request=RoiRequest(self.identity,0,(-.5,-.5,447.5,447.5),np.eye(3),source_pixel_space_id='native-v1')
    def pred(self,mask=None,identity=None,**kwargs):
        mask=self.mask if mask is None else mask;identity=self.identity if identity is None else identity
        defaults=dict(assigned_instance_id=identity.instance_id,origin='synthetic_fixture',sam_binding=dict(session_id='synthetic-session',obj_id='1',generation=0,chosen_mask_id='fixture-mask',match_basis='synthetic-independent-support'))
        defaults.update(kwargs)
        return NativeSamPrediction(prediction_metadata(self.frame,NativeMaskIdentity(self.frame.image_id,identity.instance_id,self.frame.pixel_space_id),mask,self.sources,**defaults),mask)
    def nk(self,identity=None):
        return native_key(NativeMaskIdentity('image-A',(identity or self.identity).instance_id,'native-v1'))
    def build(self,pred=None,requests=None,**kwargs):
        predictions={} if pred is None else {self.nk():pred}
        return prepare_sam_roi_inputs([self.frame],requests or [self.request],predictions,expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True,**kwargs)
    def test_rgb_mask_unit_and_negative_padding_alignment(self):
        for bounds in [(-.5,-.5,447.5,447.5),(-20.5,-10.5,427.5,437.5)]:
            request=RoiRequest(self.identity,0,bounds,source_pixel_space_id='native-v1')
            result=self.build(self.pred(),requests=[request]);red=result.rgb[0,0]*.229+.485
            self.assertTrue(torch.allclose(red,result.mask_aux.visible[0,0],atol=1e-5))
            reference,mats=prepare_rois([self.frame.pixels],[bounds],[0],roi_size=448,expansion=1.)
            self.assertTrue(torch.equal(result.rgb,reference));np.testing.assert_array_equal(result.A448[0],mats[0])
            if bounds[0]<-1:
                self.assertEqual(float(result.valid_image_support[0,0,:10].sum()),0)
                self.assertEqual(float(result.valid_image_support[0,0,:,:20].sum()),0)
                self.assertEqual(float(result.mask_aux.visible[0,0,:,:20].sum()),0)
    def test_112_224_maps_land_on_same_rgb_and_mask(self):
        request=RoiRequest(self.identity,0,(40.25,20.25,264.25,244.25),np.eye(3),source_pixel_space_id='native-v1');r=self.build(self.pred(),requests=[request]);point=np.array([[120.,110.]])
        for size,matrix in [(448,r.A448[0]),(224,r.A224[0]),(112,r.A112[0])]:
            point_n=DataAdapter.transform_points(point,matrix)
            up,_=DataAdapter.letterbox_affine((size,size),(448,448))
            uv=DataAdapter.transform_points(point_n,up)[0];direct=DataAdapter.transform_points(point,r.A448[0])[0]
            np.testing.assert_allclose(uv,direct,atol=1e-5)
            x,y=np.rint(uv).astype(int);self.assertEqual(float(r.mask_aux.visible[0,0,y,x]),1.)
            self.assertAlmostEqual(float(r.rgb[0,0,y,x]*.229+.485),1.,places=5)
        np.testing.assert_allclose(r.K112[0],r.A112[0]);np.testing.assert_allclose(r.K224[0],r.A224[0])
    def test_distance_matches_existing_maskaux_contract(self):
        r=self.build(self.pred());expected=prepare_mask_aux(r.mask_aux.visible,r.mask_aux.available,identities=(self.identity,),candidate_overlap=r.mask_aux.candidate_overlap)
        self.assertTrue(torch.equal(r.mask_aux.distance,expected.distance));self.assertGreater(float(r.mask_aux.distance[0,0,90,120]),0);self.assertLess(float(r.mask_aux.distance[0,0,10,10]),0)
    def test_missing_ambiguous_assignment_dropout(self):
        for pred,expected in [(None,'missing_prediction_cache'),(self.pred(status='ambiguous',reason='merged_support'),'ambiguous:merged_support'),(self.pred(assigned_instance_id='right'),'wrong_assigned_instance')]:
            r=self.build(pred);self.assertEqual(r.reasons[0],expected);self.assertEqual(float(r.mask_aux.available.sum()),0);self.assertEqual(float(r.mask_aux.visible.sum()),0);self.assertEqual(float(r.mask_aux.distance.sum()),0)
        r=self.build(self.pred(),dropout=torch.tensor([True]));self.assertEqual(r.reasons[0],'prediction_dropout');self.assertEqual(float(r.mask_aux.available.sum()),0)
    def test_empty_prediction_is_unavailable_not_negative(self):
        r=self.build(self.pred(mask=np.zeros_like(self.mask)));self.assertEqual(r.reasons[0],'empty_native_mask');self.assertEqual(float(r.mask_aux.available.sum()),0)
    def test_two_instances_are_keyed_and_swaps_rejected(self):
        right=FeatureIdentity('image-A','right','roi-right','roi448');other=np.zeros_like(self.mask);other[250:330,280:390]=True
        requests=[RoiRequest(right,0,self.request.bounds_native,source_pixel_space_id='native-v1'),self.request]
        left_pred=self.pred();right_pred=self.pred(mask=other,identity=right)
        mapping={self.nk():left_pred,self.nk(right):right_pred}
        r=prepare_sam_roi_inputs([self.frame],requests,mapping,expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True)
        self.assertEqual(r.mask_aux.identities,(right,self.identity));self.assertTrue(np.array_equal(r.mask_aux.visible[0,0].numpy().astype(bool),other));self.assertTrue(np.array_equal(r.mask_aux.visible[1,0].numpy().astype(bool),self.mask))
        mapping[self.nk(right)]=left_pred
        with self.assertRaisesRegex(ValueError,'identity|mismatch'):prepare_sam_roi_inputs([self.frame],requests,mapping,expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True)
    def test_wrong_image_variant_source_rejected(self):
        for field,value in [('source_sha256','c'*64),('variant_id','altered'),('variant_sha256','d'*64),('native_shape',[100,100])]:
            p=self.pred();meta=copy.deepcopy(p.metadata);meta[field]=value;meta['cache_key']=cache_key(meta)
            with self.assertRaises(ValueError):self.build(NativeSamPrediction(meta,p.mask))
        p=self.pred();meta=copy.deepcopy(p.metadata);meta['sources']['sam_model_sha256']='c'*64;meta['cache_key']=cache_key(meta)
        with self.assertRaisesRegex(ValueError,'source mismatch'):self.build(NativeSamPrediction(meta,p.mask))
    def test_unknown_cache_identity_rejected(self):
        p=self.pred();meta=copy.deepcopy(p.metadata);meta['schema']='unknown';meta['cache_key']=cache_key(meta)
        with self.assertRaises(ValueError):self.build(NativeSamPrediction(meta,p.mask))
        sources=dict(self.sources,sam_model_sha256='0'*64)
        with self.assertRaises(ValueError):prepare_sam_roi_inputs([self.frame],[self.request],{},expected_sources=sources,mask_aux_factory=MaskAux)
    def test_native_shape_nonfinite_and_content_hash(self):
        with self.assertRaises(ValueError):self.build(self.pred(mask=np.zeros((12,12),bool)))
        with self.assertRaises(ValueError):self.build(self.pred(mask=np.full((448,448),np.nan)))
        p=self.pred();changed=p.mask.copy();changed[0,0]=True
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.build(NativeSamPrediction(p.metadata,changed))
        with self.assertRaises(ValueError):sample_native_nearest(self.mask,np.full((3,3),np.nan))
    def test_groundtruth_cannot_enter_prediction_interface(self):
        with self.assertRaises(ValueError):self.build(self.pred(origin='ground_truth'))
        with self.assertRaises(ValueError):prepare_sam_roi_inputs([self.frame],[self.request],{self.nk():self.pred()},expected_sources=self.sources,mask_aux_factory=MaskAux)
    def test_quality_targets_separate_and_do_not_change_inputs(self):
        r=self.build(self.pred());before=r.mask_aux.visible.clone()
        labels={identity_key(self.identity):dict(identity=self.identity,source_pixel_space_id='native-v1',supervised=True,native_mask=np.zeros_like(self.mask),native_shape=[448,448])}
        gt=prepare_quality_targets(r,labels);self.assertTrue(bool(gt.supervised[0]));self.assertEqual(float(gt.gt_visible448.sum()),0);self.assertTrue(torch.equal(before,r.mask_aux.visible))
        unknown=prepare_quality_targets(r,{});self.assertFalse(bool(unknown.supervised[0]));self.assertTrue(torch.isnan(unknown.gt_visible448).all())
        labels[identity_key(self.identity)]['native_shape']=[16,16]
        with self.assertRaises(ValueError):prepare_quality_targets(r,labels)
    def test_quality_identity_swap_rejected(self):
        r=self.build(self.pred());other=FeatureIdentity('image-A','right','roi-right','roi448')
        with self.assertRaises(ValueError):prepare_quality_targets(r,{identity_key(self.identity):dict(identity=other,source_pixel_space_id='native-v1',supervised=True,native_mask=self.mask,native_shape=[448,448])})
    def test_actual_rgb_variant_digest_is_checked(self):
        pixels=self.frame.pixels.copy();pixels[0,0,0]=1
        wrong=ImageFrame('image-A',pixels,'native-v1','b'*64,'original',self.frame.variant_sha256)
        with self.assertRaisesRegex(ValueError,'ActualRGB'):prepare_sam_roi_inputs([wrong],[self.request],{},expected_sources=self.sources,mask_aux_factory=MaskAux)
    def test_fractional_negative_roi_checker_agrees_with_sampled_rgb_coordinates(self):
        yy,xx=np.mgrid[:64,:64];mask=((xx%3)==0)&((yy%2)==0)
        pixels=np.stack((xx*4,yy*4,mask*255),axis=-1).astype(np.uint8)
        frame=ImageFrame('image-A',pixels,'native-v1','b'*64,'coordinate_fixture',array_digest(pixels))
        req=RoiRequest(self.identity,0,(-3.25,-7.75,60.75,56.25),source_pixel_space_id='native-v1')
        meta=prediction_metadata(frame,NativeMaskIdentity(frame.image_id,self.identity.instance_id,frame.pixel_space_id),mask,self.sources,assigned_instance_id='left',origin='synthetic_fixture',sam_binding=dict(session_id='fixture',obj_id='1',generation=0,chosen_mask_id='checker',match_basis='fixture'))
        r=prepare_sam_roi_inputs([frame],[req],{self.nk():NativeSamPrediction(meta,mask)},expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True)
        x=(r.rgb[0,0].numpy()*.229+.485)*255/4;y=(r.rgb[0,1].numpy()*.224+.456)*255/4
        # Use the bilinearly sampled coordinate-coded RGB as the independent
        # coordinate witness; exclude only image-edge mixing and exact nearest ties.
        interior=(x>1)&(x<62)&(y>1)&(y<62)&(np.abs((x%1)-.5)>1e-3)&(np.abs((y%1)-.5)>1e-3)
        expected=((np.rint(x).astype(int)%3)==0)&((np.rint(y).astype(int)%2)==0)
        actual=r.mask_aux.visible[0,0].numpy().astype(bool)
        self.assertGreater(int(interior.sum()),100000)
        self.assertTrue(np.array_equal(actual[interior],expected[interior]))

    def test_cache_file_is_not_accepted_by_existence_only(self):
        folder=Path('/data/zy/vln/a-test/outputs/action4_sam_roi_preparation_v001/cache_fixtures');folder.mkdir(parents=True,exist_ok=True);p=self.pred();filename=folder/'synthetic_prediction.npz'
        np.savez_compressed(filename,metadata=np.array(json.dumps(p.metadata)),mask=p.mask)
        loaded=read_prediction_cache(filename);r=self.build(loaded);self.assertEqual(float(r.mask_aux.available.sum()),1.)
        changed=p.mask.copy();changed[0,0]=True;np.savez_compressed(filename,metadata=np.array(json.dumps(p.metadata)),mask=changed)
        with self.assertRaises(ValueError):self.build(read_prediction_cache(filename))

    def test_one_verified_native_prediction_reused_across_two_jitter_rois(self):
        from unittest.mock import patch
        frame=verify_image_frame(self.frame)
        nid=NativeMaskIdentity('image-A','left','native-v1')
        pred=verify_native_prediction(self.pred(),frame,nid,self.sources,allow_synthetic=True)
        a=FeatureIdentity('image-A','left','crop-a','roi448');b=FeatureIdentity('image-A','left','crop-b','roi448')
        requests=[RoiRequest(a,0,(-.5,-.5,447.5,447.5),source_pixel_space_id='native-v1'),RoiRequest(b,0,(40.5,30.5,264.5,254.5),source_pixel_space_id='native-v1')]
        with patch('data.action4_sam_roi_inputs.array_digest',side_effect=AssertionError('Full bytehash repeated afterverification')):
            r=prepare_sam_roi_inputs([frame],requests,{native_key(nid):pred},expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True)
        self.assertEqual(r.mask_aux.identities,(a,b));self.assertEqual(float(r.mask_aux.available.sum()),2.)
        self.assertNotEqual(r.source_dependencies[0]['A448_sha256'],r.source_dependencies[1]['A448_sha256'])
        self.assertFalse(np.array_equal(r.A448[0],r.A448[1]));self.assertGreater(float(r.mask_aux.visible[1].sum()),float(r.mask_aux.visible[0].sum()))
        self.assertEqual(r.source_dependencies[0]['native_cache_identity'],r.source_dependencies[1]['native_cache_identity'])
        self.assertEqual(r.source_dependencies[0]['source_pixel_space_id'],'native-v1');self.assertEqual(r.source_dependencies[0]['roi_pixel_space_id'],'roi448')
        self.assertNotIn('crop-a',pred.metadata_json);self.assertNotIn('crop-b',pred.metadata_json)
        labels={identity_key(i):dict(identity=i,source_pixel_space_id='native-v1',supervised=True,native_mask=self.mask,native_shape=[448,448]) for i in (a,b)}
        gt=prepare_quality_targets(r,labels);self.assertEqual(gt.gt_identities,(a,b));self.assertTrue(torch.equal(gt.gt_visible448,r.mask_aux.visible))
    def test_source_and_output_pixel_domains_cannot_be_conflated(self):
        wrong=RoiRequest(self.identity,0,self.request.bounds_native,source_pixel_space_id='wrong-native')
        with self.assertRaises(ValueError):self.build(self.pred(),requests=[wrong])
        mislabeled=FeatureIdentity('image-A','left','roi-left','native-v1')
        with self.assertRaisesRegex(ValueError,'Output ROI'):self.build(self.pred(),requests=[RoiRequest(mislabeled,0,self.request.bounds_native,source_pixel_space_id='native-v1')])
        r=self.build(self.pred());labels={identity_key(self.identity):dict(identity=self.identity,source_pixel_space_id='other-native',supervised=True,native_mask=self.mask,native_shape=[448,448])}
        with self.assertRaisesRegex(ValueError,'source pixel-space'):prepare_quality_targets(r,labels)
    def test_verified_arrays_are_bytes_backed_and_isolated_from_original_mutation(self):
        frame=verify_image_frame(self.frame);nid=NativeMaskIdentity('image-A','left','native-v1');pred=verify_native_prediction(self.pred(),frame,nid,self.sources,allow_synthetic=True)
        with self.assertRaises(ValueError):frame.pixels.setflags(write=True)
        with self.assertRaises(ValueError):pred.mask.setflags(write=True)
        self.frame.pixels[:]=255;self.mask[:]=False
        r=prepare_sam_roi_inputs([frame],[self.request],{native_key(nid):pred},expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True)
        self.assertGreater(float(r.mask_aux.visible.sum()),0);self.assertNotEqual(int(frame.pixels[0,0,0]),255)
        pred.metadata['native_identity'][0]='mutated-copy'
        self.assertEqual(pred.metadata['native_identity'][0],'image-A')
    def test_verified_array_shape_tampering_is_rejected(self):
        frame=verify_image_frame(self.frame);nid=NativeMaskIdentity('image-A','left','native-v1');pred=verify_native_prediction(self.pred(),frame,nid,self.sources,allow_synthetic=True)
        pred.mask.shape=(224,896)
        with self.assertRaisesRegex(ValueError,'mutable verified mask'):prepare_sam_roi_inputs([frame],[self.request],{native_key(nid):pred},expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True)
        frame.pixels.shape=(224,896,3)
        with self.assertRaisesRegex(ValueError,'verified frame'):prepare_sam_roi_inputs([frame],[self.request],{},expected_sources=self.sources,mask_aux_factory=MaskAux)

    def test_upstream_failure_and_visibility_statuses_remain_distinct_unavailable(self):
        from model.action4_sam_feature_adapter import SamFeatureOutput,mask_quality_loss
        for status in ('target_visibility_unknown','inference_error','missing_response','session_lost'):
            with self.subTest(status=status):
                r=self.build(self.pred(status=status,reason='upstream:'+status))
                self.assertEqual(r.reasons[0],status+':upstream:'+status)
                self.assertEqual(r.source_dependencies[0]['upstream_status'],status)
                self.assertEqual(r.source_dependencies[0]['upstream_reason'],'upstream:'+status)
                self.assertEqual(r.source_dependencies[0]['cache_metadata']['status'],status)
                self.assertFalse(r.source_dependencies[0]['use_as_training_negative'])
                self.assertEqual(float(r.mask_aux.available.sum()),0.)
                self.assertEqual(float(r.mask_aux.visible.sum()),0.);self.assertEqual(float(r.mask_aux.distance.sum()),0.)
                aux=r.mask_aux
                output=SamFeatureOutput(torch.empty(0),torch.empty(0),torch.tensor([[.5]],requires_grad=True),torch.zeros(1,1),aux.visible,aux.available,aux.dropout,aux.identities)
                loss=mask_quality_loss(output,torch.zeros_like(aux.visible),gt_identities=aux.identities,supervised=torch.ones(1,dtype=torch.bool))
                self.assertEqual(int(loss.count),0);self.assertEqual(float(loss.loss_sum.detach()),0.)
    def test_only_canonical_visibility_token_and_explicit_reason_accepted(self):
        with self.assertRaisesRegex(ValueError,'Unknown predicted mask status'):
            self.build(self.pred(status='visibility_unknown',reason='unregistered alias'))
        with self.assertRaisesRegex(ValueError,'requires reason'):
            self.build(self.pred(status='target_visibility_unknown',reason=None))

    def test_verified_frame_replace_cannot_inherit_verification(self):
        from dataclasses import replace
        frame=verify_image_frame(self.frame)
        other=np.frombuffer(np.zeros_like(frame.pixels).tobytes(),dtype=np.uint8).reshape(frame.pixels.shape)
        for changed in (replace(frame,pixels=other),replace(frame,variant_id='different-variant')):
            self.assertIsNone(changed._token)
            with self.assertRaisesRegex(ValueError,'verified frame'):
                prepare_sam_roi_inputs([changed],[self.request],{},expected_sources=self.sources,mask_aux_factory=MaskAux)
    def test_verified_mask_replace_cannot_inherit_verification(self):
        from dataclasses import replace
        frame=verify_image_frame(self.frame);nid=NativeMaskIdentity('image-A','left','native-v1')
        pred=verify_native_prediction(self.pred(),frame,nid,self.sources,allow_synthetic=True)
        other=np.frombuffer(np.zeros_like(pred.mask).tobytes(),dtype=np.bool_).reshape(pred.mask.shape)
        metadata=pred.metadata;metadata['reason']='changed';metadata['cache_key']=cache_key(metadata)
        for changed in (replace(pred,mask=other),replace(pred,metadata_json=json.dumps(metadata)),replace(pred,has_foreground=False)):
            self.assertIsNone(changed._token)
            with self.assertRaisesRegex(ValueError,'verified mask'):
                prepare_sam_roi_inputs([frame],[self.request],{native_key(nid):changed},expected_sources=self.sources,mask_aux_factory=MaskAux,allow_synthetic=True)
    def test_verified_cache_is_process_local_and_not_pickleable(self):
        import pickle,copy
        from unittest.mock import patch
        frame=verify_image_frame(self.frame);nid=NativeMaskIdentity('image-A','left','native-v1')
        pred=verify_native_prediction(self.pred(),frame,nid,self.sources,allow_synthetic=True)
        for obj in (frame,pred):
            with self.assertRaisesRegex(TypeError,'process-local'):pickle.dumps(obj)
            with self.assertRaisesRegex(TypeError,'process-local'):copy.copy(obj)
        with patch('data.action4_sam_roi_inputs.os.getpid',return_value=-1):
            with self.assertRaisesRegex(ValueError,'verified frame'):
                prepare_sam_roi_inputs([frame],[self.request],{},expected_sources=self.sources,mask_aux_factory=MaskAux)

if __name__=='__main__':unittest.main()
