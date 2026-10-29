"""Analytic rigid-cuboid decoder fixtures; no trained accuracy claim."""
import copy,unittest
import numpy as np
from model.action4_single_frame_pose import POLICY,decode_single_frame_pose,observations_from_head

class SingleFramePoseTests(unittest.TestCase):
    def setUp(self):
        self.cad=np.array([[-.035,-.016,-.022],[.035,-.016,-.022],[.035,.016,-.022],[-.035,.016,-.022],
                           [-.035,-.016,.022],[.035,-.016,.022],[.035,.016,.022],[-.035,.016,.022]])
        self.K=np.array([[1000.,0,320],[0,1000.,240],[0,0,1]])
        self.A=np.array([[1.4,0,-336],[0,1.4,-224],[0,0,1.]])
        self.truth=np.array([0.,0.,.7]);self.raw=np.array([.008,0.,.7]);self.identity=("frameA","instanceA","roiA","roi448")
        self.uv=self.project(self.truth)
        self.box=np.r_[self.uv.min(0),self.uv.max(0)]
    def project(self,t):
        h=(self.cad+t)@self.K.T
        return h[:,:2]/h[:,2:]
    def call(self,*,uv=None,A=None,b=None,quality=True,diagnostic=False,route=None,raw=None,mutate=None,raw_identity=None,roi_identity=None):
        uv=self.uv if uv is None else uv;A=self.A if A is None else A
        obs={'identity':self.identity,'corner_ids':list(range(8)),'uv224':(np.c_[uv,np.ones(8)]@A.T)[:,:2],
             'p_valid':np.ones(8)*.99,'q_corner':np.ones(8)*.95,'b224':np.ones(8) if b is None else b,
             'multimodal':np.zeros(8,dtype=bool),'pixel_space_id':'analytic_native'}
        if mutate:mutate(obs)
        if route is not None:
            route=dict(route);route.setdefault("identity",self.identity)
        return decode_single_frame_pose(canonical_m=self.cad,K_native=self.K,A224_from_native=A,
            raw_rotation=np.eye(3),raw_translation_m=self.raw if raw is None else raw,
            observations=obs,route=route or {'kind':'full','pixel_space_id':'analytic_native','full_box_quality_passed':True,'full_box_native':self.box,'identity':self.identity},
            pixel_space_id='analytic_native',raw_identity=self.identity if raw_identity is None else raw_identity,roi_identity=self.identity if roi_identity is None else roi_identity,quality_ready=quality,config=copy.deepcopy(POLICY),diagnostic_candidate=diagnostic)
    def test_rigid_refinement_consumes_changed_observations(self):
        a=self.call();self.assertTrue(a['accepted'],a)
        self.assertLess(a['candidate_corner_objective'],.9*a['raw_corner_objective'])
        self.assertLess(np.linalg.norm(np.asarray(a['translation_m'])-self.truth),np.linalg.norm(self.raw-self.truth))
        b=self.call(uv=self.uv+np.array([2.,0.]));self.assertTrue(b['accepted'],b)
        self.assertGreater(np.linalg.norm(np.asarray(a['translation_m'])-b['translation_m']),1e-5)
        self.assertLessEqual(sum(c['iterations'] for c in a['candidates']),10)
    def test_quality_not_ready_keeps_raw_even_with_good_candidate(self):
        a=self.call(quality=False);self.assertFalse(a['accepted']);self.assertEqual(a['translation_m'],self.raw.tolist())
        b=self.call(quality=False,diagnostic=True);self.assertTrue(b['geometric_candidate_accepted'],b)
        self.assertFalse(b['accepted']);self.assertEqual(b['translation_m'],self.raw.tolist())
    def test_insufficient_multimodal_and_domain(self):
        a=self.call(mutate=lambda o:o['p_valid'].__setitem__(slice(0,3),.1));self.assertEqual(a['reason'],'insufficient_reliable_corner_ids')
        b=self.call(mutate=lambda o:o['multimodal'].__setitem__(slice(0,3),True));self.assertEqual(b['reason'],'insufficient_reliable_corner_ids')
        c=self.call(mutate=lambda o:o['uv224'].__setitem__(slice(0,3),[-1.,50.]));self.assertEqual(c['reason'],'insufficient_reliable_corner_ids')
    def test_pixel_scale_transport_equivalence(self):
        a=self.call()
        A=np.array([[.7,0,-112],[0,.7,-56],[0,0,1.]])
        b=self.call(A=A,b=np.ones(8)*.5)
        self.assertTrue(a['accepted'] and b['accepted'])
        np.testing.assert_allclose(a['translation_m'],b['translation_m'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(a['rotation'],b['rotation'],atol=1e-6,rtol=0)
    def test_prediction_route_protection_and_near_depth(self):
        bad={'kind':'full','pixel_space_id':'analytic_native','full_box_quality_passed':False,'full_box_native':self.box}
        self.assertEqual(self.call(route=bad)['reason'],'full_box_not_admitted')
        bad['full_box_quality_passed']=True;bad['full_box_native']=[0,0,20,20]
        self.assertEqual(self.call(route=bad)['reason'],'full_box_support_gate')
        self.assertFalse(self.call(raw=np.array([0,0,-1.]))['pose_valid'])
        self.assertFalse(self.call(raw=self.truth)['accepted']) # zero residual is not an improvement
    def test_coarse_support_and_invalid_contracts(self):
        m=np.zeros((480,640),bool);m[230:250,310:330]=True
        route={'kind':'coarse','pixel_space_id':'analytic_native','support_mask_native':m,'candidate_bounds_native':[200,100,440,380]}
        self.assertTrue(self.call(route=route)['accepted'])
        m[:]=False;m[:5,:5]=True
        self.assertEqual(self.call(route=route)['reason'],'coarse_visible_support_gate')
        for edit in (lambda o:o['corner_ids'].__setitem__(0,True),lambda o:o['corner_ids'].reverse(),
                     lambda o:o['uv224'].__setitem__((0,0),np.nan)):
            with self.assertRaises(ValueError):self.call(mutate=edit)
    def test_raw_roi_and_route_identity_swaps_rejected(self):
        wrong=("frameB","instanceA","roiA","roi448")
        with self.assertRaisesRegex(ValueError,"identity"):self.call(raw_identity=wrong)
        with self.assertRaisesRegex(ValueError,"identity"):self.call(roi_identity=wrong)
        route={"kind":"full","pixel_space_id":"analytic_native","full_box_quality_passed":True,"full_box_native":self.box,"identity":wrong}
        with self.assertRaisesRegex(ValueError,"identity"):self.call(route=route)

    def test_head_multipeak_and_identity_transport(self):
        import torch
        from model.action4_sam_feature_adapter import FeatureIdentity
        from model.action4_corner_observation_head import CornerLocalizationOutput
        from model.action4_corner_supervision import CornerObservationOutput
        torch.set_num_threads(1)
        ident=FeatureIdentity('fixture','instance','roi','roi448')
        h=torch.full((1,8,224,224),-20.);h[:,:,100,100]=0.;h[:,0,130,130]=0.
        output=CornerObservationOutput(CornerLocalizationOutput(h,(ident,)),torch.ones(1,8,2)*100,
                torch.ones(1,8)*5,torch.ones(1,8),torch.ones(1,8)*5)
        obs=observations_from_head(output,0,identity=ident,pixel_space_id='analytic_native',config=copy.deepcopy(POLICY))
        self.assertEqual(np.flatnonzero(obs['multimodal']).tolist(),[0])
        # A locally valid typed B observation must not modify A's raw pose/ROI.
        for key in range(3):
            changed=list(self.identity);changed[key]+="_different"
            wrong=FeatureIdentity(*changed)
            source=CornerObservationOutput(CornerLocalizationOutput(h,(wrong,)),output.uv224,
                output.valid_logits,output.scale_b224,output.quality_logits)
            swapped=observations_from_head(source,0,identity=wrong,pixel_space_id='analytic_native',config=POLICY)
            with self.assertRaisesRegex(ValueError,"identity"):
                self.call(mutate=lambda obs:obs.update(swapped))
        other=FeatureIdentity('fixture','wrong','roi','roi448')
        with self.assertRaises(ValueError):observations_from_head(output,0,identity=other,pixel_space_id='analytic_native',config=POLICY)

if __name__=='__main__':unittest.main()
