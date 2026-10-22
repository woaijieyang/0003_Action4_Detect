"""Pure CPU synthetic fixtures; no project sources/assets, GPU or formal records."""
from copy import deepcopy
import unittest
import numpy as np
from src.data import action4_parameter_sampling as n


def fixtures():
    corners=np.array([[-.04,-.025,-.015],[.04,-.025,-.015],[.04,.025,-.015],[-.04,.025,-.015],
                      [-.04,-.025,.015],[.04,-.025,.015],[.04,.025,.015],[-.04,.025,.015]])
    A=np.eye(4);A[:3,:3]=n.rotation_xyz([3.,-5.,7.]);A[:3,3]=[.013,-.007,.009]
    anchor={'schema':'action4.modular_pose_controls.v1','K_native':[[1700.,0.,1624.],[0.,1700.,1232.],[0.,0.,1.]],
            'W_from_capture':np.eye(4).tolist(),'sides':{},'actor_pose_capture_by_side':{},
            'environment_controls':{'wall_visible':True,'table_visible':True,'area_light_gain':1.,'exposure_offset_ev':0.},
            'provenance':{'source_context_sha256':'a'*64}}
    for side,x in [('left',-.12),('right',.12)]:
        B=np.eye(4);B[:3,3]=[x,0.,.6]
        anchor['sides'][side]={key:(A if key=='A_brace_from_CAD' else np.eye(4)).tolist() for key in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace')}
        anchor['actor_pose_capture_by_side'][side]={'brace_B_capture':B.tolist(),'core_C_capture':(B@A).tolist(),'whole_actor_delta_capture':np.eye(4).tolist()}
    entry={'sample_id':'synthetic_fixture_only','index':0,'seed':2026,'scene_family':'wall_rear',
           'target_count':2,'target_presence':[True,True],'group_id':'fixture_group','background_source_id':'fixture_background',
           'background_seed':17,'split':'train'}
    policy=n.default_policy(corners,{'path':'synthetic_fixture_only/canonical.json','sha256':'1'*64})
    ref={'path':'synthetic_fixture_only/anchor.json','sha256':'2'*64}
    return anchor,entry,policy,corners,ref


class NumericTests(unittest.TestCase):
    def propose(self,vector=None,**overrides):
        anchor,entry,policy,corners,ref=fixtures()
        return n.propose(overrides.get('anchor',anchor),overrides.get('entry',entry),overrides.get('policy',policy),
                         vector or n.latin_hypercube(16,17)[0],overrides.get('corners',corners),anchor_ref=ref,diagnostics=overrides.get('diagnostics',True))

    def test_fixed_mount_dimensions_and_exact_centres(self):
        a,e,p,c,r=fixtures();before=n.digest(a)
        result=n.propose(a,e,p,n.latin_hypercube(16,17)[3],c,anchor_ref=r)
        self.assertEqual(n.digest(a),before)
        self.assertEqual(result['schema'],'action4.modular_pose_controls.v2')
        self.assertFalse(result['training_accepted']);self.assertFalse(result['dataset_generation_allowed'])
        self.assertFalse(result['provenance']['source_file_identities_verified'])
        expected=np.linalg.norm(c[:,None,:]-c[None,:,:],axis=-1)
        for side in n.SIDES:
            actor=result['actor_pose_capture_by_side'][side];B=np.array(actor['brace_B_capture']);C=np.array(actor['core_C_capture']);A=np.array(a['sides'][side]['A_brace_from_CAD'])
            np.testing.assert_allclose(B@A,C,rtol=0,atol=1e-12)
            for key in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace'):
                self.assertEqual(result['sides'][side][key],a['sides'][side][key])
            actual=np.array(result['derived']['by_side'][side]['canonical8_capture_m'])
            np.testing.assert_allclose(np.linalg.norm(actual[:,None,:]-actual[None,:,:],axis=-1),expected,rtol=0,atol=1e-12)
            np.testing.assert_allclose(result['derived']['by_side'][side]['projected_body_centre_fraction'],result['controls']['requested_by_side'][side]['centre_fraction'],rtol=0,atol=1e-12)
            self.assertEqual(result['derived']['by_side'][side]['corner_ids'],list(range(8)))
            self.assertFalse(result['derived']['by_side'][side]['points_clamped'])
            self.assertEqual(result['controls']['requested_by_side'][side]['installation_angle_offset_deg'],[0.,0.,0.])

    def test_independent_left_right_and_depth(self):
        sample=n.latin_hypercube(1,31)[0];changed=deepcopy(sample)
        j=n.DIMENSIONS.index('left.u');changed['values'][j]=(sample['values'][j]+.5)%1
        a=self.propose(sample);b=self.propose(changed)
        self.assertEqual(a['actor_pose_capture_by_side']['right'],b['actor_pose_capture_by_side']['right'])
        self.assertNotEqual(a['actor_pose_capture_by_side']['left'],b['actor_pose_capture_by_side']['left'])
        self.assertNotEqual(a['derived']['relative_pair'],b['derived']['relative_pair'])
        changed=deepcopy(sample);j=n.DIMENSIONS.index('right.depth');changed['values'][j]=(sample['values'][j]+.5)%1
        b=self.propose(changed)
        self.assertEqual(a['actor_pose_capture_by_side']['left'],b['actor_pose_capture_by_side']['left'])
        self.assertNotEqual(a['derived']['by_side']['right']['depth_m'],b['derived']['by_side']['right']['depth_m'])

    def test_lhs_every_stratum_and_reproducibility(self):
        a=n.latin_hypercube(32,19);b=n.latin_hypercube(32,19)
        self.assertEqual(a,b);self.assertNotEqual(a,n.latin_hypercube(32,20))
        for j in range(len(n.DIMENSIONS)):
            self.assertEqual(sorted(x['stratum_ids'][j] for x in a),list(range(32)))
        self.assertEqual(self.propose(a[2]),self.propose(b[2]))
        self.assertNotEqual(self.propose(a[2])['geometry_descriptor'],self.propose(a[3])['geometry_descriptor'])

    def test_order_mutation_and_nonfinite_rejected(self):
        a,e,p,c,r=fixtures();c[[0,1]]=c[[1,0]]
        with self.assertRaises(ValueError):self.propose(corners=c)
        a,e,p,c,r=fixtures();c[0,0]=np.nan
        with self.assertRaises(ValueError):self.propose(corners=c)
        a['sides']['left']['A_brace_from_CAD'][0][0]*=2
        with self.assertRaises(ValueError):self.propose(anchor=a)

    def test_bad_domain_vector_presence_and_anchor(self):
        a,e,p,c,r=fixtures();p['depth_m']=[-.1,.9]
        with self.assertRaises(ValueError):self.propose(policy=p)
        a,e,p,c,r=fixtures();p['training_accepted']=True
        with self.assertRaises(ValueError):self.propose(policy=p)
        s=n.latin_hypercube(16,19)[0];s['values'][0]=1.
        with self.assertRaises(ValueError):self.propose(s)
        a,e,p,c,r=fixtures();e['target_count']=1
        with self.assertRaises(ValueError):self.propose(entry=e)
        a,e,p,c,r=fixtures();a['actor_pose_capture_by_side']['left']['core_C_capture'][0][3]+=1
        with self.assertRaises(ValueError):self.propose(anchor=a)

    def test_invalid_reference_and_nonrigid_source_rejected(self):
        for path in ('../source.json','/outside/source.json','C:/outside.json','a//b.json'):
            with self.assertRaises(ValueError):n.reference({'path':path,'sha256':'a'*64})
        a,e,p,c,r=fixtures();a['W_from_capture'][1][1]=2
        with self.assertRaises(ValueError):self.propose(anchor=a)
        s=n.latin_hypercube(2,5)[0];s['stratum_ids'][0]=1-s['stratum_ids'][0]
        with self.assertRaises(ValueError):self.propose(s)

    def test_actual_corner_near_plane_is_rejected_prior_not_accepted(self):
        a,e,p,c,r=fixtures();p['depth_m']=[.005,.005];p['brace_local_rotation_xyz_deg']={k:[0.,0.] for k in ('rx','ry','rz')}
        value=self.propose(policy=p)
        self.assertEqual(value['status'],'NUMERICAL_PROPOSAL_REJECTED_BY_PRIOR')
        self.assertFalse(value['safety_prior']['eligible_for_actual_checks'])
        self.assertIsNone(value['derived']['by_side']['left']['canonical8_native_px'])

    def test_camera_envelopes_never_claim_actual_collision_pass(self):
        a,e,p,c,r=fixtures();I=np.eye(4);other=I.copy();other[0,3]=1
        separated=n.envelope_prior(c,I,other)
        self.assertTrue(separated['canonical_envelopes_disjoint']);self.assertFalse(separated['actual_surface_pass'])
        overlap=n.envelope_prior(c,I,I)
        self.assertFalse(overlap['canonical_envelopes_disjoint']);self.assertFalse(overlap['overlap_means_actual_collision'])

    def test_absent_has_no_target_labels_or_occlusion(self):
        a,e,p,c,r=fixtures();e.update(target_count=1,target_presence=[True,False])
        result=self.propose(entry=e)
        self.assertIsNone(result['sides']['right']['core_C_capture'])
        self.assertIsNone(result['derived']['by_side']['right'])
        self.assertEqual(set(result['occlusion_condition']['by_instance_id']),{'1'})
        self.assertNotEqual(result['actor_pose_capture_by_side']['right'],a['actor_pose_capture_by_side']['right'])
        self.assertIn('2',result['geometry_descriptor']['absent_actor_only_quantized'])
        e.update(target_count=0,target_presence=[False,False],scene_family='hard_negative')
        result=self.propose(entry=e)
        self.assertEqual(result['geometry_descriptor']['instance_quantized'],{})
        self.assertEqual(result['occlusion_condition']['by_instance_id'],{})

    def test_duplicate_counts_ignore_photometry_and_frame_separate(self):
        a=self.propose();b=deepcopy(a);b['environment_controls']['area_light_gain']=1.04
        report=n.diversity_counts([a,b])
        self.assertEqual(report['frames']['count'],2);self.assertEqual(report['frames']['unique_quantized_groups'],1)
        self.assertEqual(report['instances_pooled']['count'],4)
        self.assertEqual(report['instances_by_physical_id']['1']['count'],2)
        self.assertFalse(report['actual_scene_or_mask_diversity_measured'])
        items=[self.propose(v) for v in n.latin_hypercube(16,25)]
        self.assertGreaterEqual(n.diversity_counts(items)['frames']['unique_quantized_groups'],12)

    def test_one_fixed_camera_visible_in_per_instance_coverage(self):
        sample=n.latin_hypercube(1,31)[0];changed=deepcopy(sample)
        j=n.DIMENSIONS.index('left.u');changed['values'][j]=(sample['values'][j]+.5)%1
        a=self.propose(sample);b=self.propose(changed);report=n.diversity_counts([a,b])
        self.assertEqual(report['near_projection_by_physical_id']['2']['near_projection_groups'],1)
        self.assertEqual(report['near_projection_by_physical_id']['1']['near_projection_groups'],2)
        self.assertEqual(report['near_projection_frames']['near_projection_groups'],2)
        self.assertEqual(report['coverage_by_physical_id']['2']['centre_u_fraction']['span'],0.)
        self.assertGreater(report['relative_pair_coverage']['delta_x_m']['span'],0.)
        tiny=deepcopy(sample);tiny['values'][j]+=1e-5
        self.assertEqual(n.diversity_counts([a,self.propose(tiny)])['near_projection_frames']['near_projection_groups'],1)

    def test_search_domain_is_configurable_not_twenty_degree_limit(self):
        a,e,p,c,r=fixtures();p['brace_local_rotation_xyz_deg']={k:[30.,35.] for k in ('rx','ry','rz')};p['depth_m']=[.95,1.05]
        result=self.propose(policy=p)
        self.assertGreater(result['controls']['requested_by_side']['left']['brace_local_rotation_xyz_deg'][0],20.)
        self.assertGreater(result['derived']['by_side']['left']['depth_m'],.9)
        self.assertFalse(result['ranges_actual_boundary_validated'])

    def test_joint_uniform_photometry_and_default_no_diagnostics(self):
        anchor,entry,policy,corners,ref=fixtures()
        v=n.sample_uniform(12,9)
        self.assertEqual(v,n.sample_uniform(12,9));self.assertNotEqual(v,n.sample_uniform(13,9))
        self.assertEqual(len(v['values']),28)
        result=n.propose(anchor,entry,policy,v,corners,anchor_ref=ref)
        self.assertIsNone(result['effective_changes']['from_anchor'])
        self.assertIsNone(result['effective_changes']['from_previous'])
        self.assertFalse(result['effective_changes']['computed'])
        self.assertFalse(result['safety_prior']['diagnostics_evaluated'])
        self.assertIsNone(result['safety_prior']['eligible_for_actual_checks'])
        self.assertEqual(result['safety_prior']['camera_pair']['method'],'not_evaluated')
        self.assertEqual(policy['depth_m'],[.4,1.4])
        self.assertTrue(.5<=result['environment_controls']['area_light_gain']<=1.8)
        self.assertTrue(-.7<=result['environment_controls']['exposure_offset_ev']<=.7)
        self.assertNotEqual(result['environment_controls'],anchor['environment_controls'])
        bad=deepcopy(v);bad['values'][0]+=.01
        with self.assertRaises(ValueError):n.propose(anchor,entry,policy,bad,corners,anchor_ref=ref)

    def test_opt_in_changes_and_negative_actor_diversity(self):
        anchor,entry,policy,corners,ref=fixtures()
        a=n.propose(anchor,entry,policy,n.sample_uniform(0,42),corners,anchor_ref=ref,include_change_counts=True)
        b=n.propose(anchor,entry,policy,n.sample_uniform(1,42),corners,anchor_ref=ref,previous_proposal=a,include_change_counts=True)
        self.assertGreater(a['effective_changes']['from_anchor']['changed_group_count'],5)
        self.assertGreater(b['effective_changes']['from_previous']['changed_group_count'],5)
        same=n.propose(anchor,entry,policy,n.sample_uniform(0,42),corners,anchor_ref=ref,previous_proposal=a,include_change_counts=True)
        self.assertEqual(same['effective_changes']['from_previous']['changed_group_count'],0)
        entry.update(target_count=0,target_presence=[False,False],scene_family='hard_negative')
        items=[n.propose(anchor,entry,policy,n.sample_uniform(i,42),corners,anchor_ref=ref) for i in range(4)]
        report=n.diversity_counts(items)
        self.assertEqual(report['frames']['unique_quantized_groups'],1)
        self.assertEqual(report['actor_frames_including_absent']['unique_quantized_groups'],4)
        self.assertTrue(all(x['geometry_descriptor']['instance_quantized']=={} for x in items))

    def test_normalized_projection_nearness_is_optional(self):
        anchor,entry,policy,corners,ref=fixtures();policy['descriptor']['projection_near_mode']='target_diagonal_fraction'
        a=n.propose(anchor,entry,policy,n.sample_uniform(0,1),corners,anchor_ref=ref)
        report=n.diversity_counts([a,deepcopy(a)])
        self.assertEqual(report['near_projection_frames']['near_projection_groups'],1)
        self.assertEqual(report['near_projection_frames']['comparison_mode'],'target_diagonal_fraction')

    def test_delta_is_relative_to_original_base_not_anchor(self):
        a,e,p,c,r=fixtures()
        for side in n.SIDES:
            D=np.eye(4);D[:3,3]=[.03,-.02,.01]
            a['actor_pose_capture_by_side'][side]['whole_actor_delta_capture']=D.tolist()
        result=self.propose(anchor=a)
        for side in n.SIDES:
            original_base=np.linalg.inv(np.array(a['actor_pose_capture_by_side'][side]['whole_actor_delta_capture']))@np.array(a['actor_pose_capture_by_side'][side]['brace_B_capture'])
            np.testing.assert_allclose(np.array(result['actor_pose_capture_by_side'][side]['whole_actor_delta_capture'])@original_base,result['actor_pose_capture_by_side'][side]['brace_B_capture'],rtol=0,atol=1e-12)


    def test_foreground_patch_projected_geometry_and_target_not_measurement(self):
        a,e,p,c,ref=fixtures()
        result=n.propose(a,e,p,n.sample_uniform(8,91),c,anchor_ref=ref)
        for side,item in result['occluder_proposals'].items():
            self.assertIsNone(item['actual_occlusion_fraction'])
            self.assertTrue(item['actual_application_pending'])
            q=np.array(item['vertices_capture_m']);hom=q@np.array(a['K_native']).T
            np.testing.assert_allclose(hom[:,:2]/hom[:,2,None],item['vertices_native_px'],atol=1e-9)
            self.assertTrue(np.all(q[:,2]<np.array(result['derived']['by_side'][side]['canonical8_capture_m'])[:,2].min()))
            np.testing.assert_allclose(q,item['vertices_world_m'],atol=1e-12)
            self.assertGreater(min(item['patch_dimensions_m']),0)

    def test_occluder_parameter_changes_geometry_without_moving_camera(self):
        a,e,p,c,ref=fixtures();v=n.latin_hypercube(1,90)[0]
        baseline=n.propose(a,e,p,v,c,anchor_ref=ref)
        for name in n.OCCLUSION_DIMENSIONS:
            changed=deepcopy(v);j=n.DIMENSIONS.index(name);changed['values'][j]=(changed['values'][j]+.4)%1
            result=n.propose(a,e,p,changed,c,anchor_ref=ref)
            self.assertEqual(result['actor_pose_capture_by_side'],baseline['actor_pose_capture_by_side'])
            side=name.split('.')[0];other='right' if side=='left' else 'left'
            self.assertNotEqual(result['occluder_proposals'][side]['vertices_capture_m'],baseline['occluder_proposals'][side]['vertices_capture_m'])
            self.assertEqual(result['occluder_proposals'][other],baseline['occluder_proposals'][other])

    def test_fixed_background_is_not_a_random_dimension(self):
        a,e,p,c,ref=fixtures()
        self.assertNotIn('background',n.DIMENSIONS)
        for name in p['background_templates']:
            for i in range(3):
                result=n.propose(a,e,p,n.sample_uniform(i,3),c,anchor_ref=ref,template_name=name)
                self.assertEqual(result['background_request']['template'],name)
                self.assertEqual(result['background_request']['selection'],'fixed_worker_template')
        with self.assertRaises(ValueError):n.propose(a,e,p,n.sample_uniform(0,3),c,anchor_ref=ref,template_name='unknown')

    def test_no_absent_occluder_and_nonidentity_world_conversion(self):
        a,e,p,c,ref=fixtures();W=np.eye(4);W[:3,:3]=n.rotation_xyz([20,30,-10]);W[:3,3]=[1,2,3];a['W_from_capture']=W.tolist()
        e.update(target_count=1,target_presence=[False,True])
        result=n.propose(a,e,p,n.sample_uniform(9,87),c,anchor_ref=ref)
        self.assertEqual(set(result['occluder_proposals']),{'right'})
        item=result['occluder_proposals']['right'];q=np.array(item['vertices_capture_m'])
        np.testing.assert_allclose(q@W[:3,:3].T+W[:3,3],item['vertices_world_m'],atol=1e-12)

    def test_local_axes_are_independently_configurable(self):
        a,e,p,c,ref=fixtures();p['brace_local_rotation_xyz_deg']={'rx':[35,35],'ry':[-60,-60],'rz':[50,50]}
        result=n.propose(a,e,p,n.sample_uniform(3,99),c,anchor_ref=ref)
        self.assertEqual(result['controls']['requested_by_side']['left']['brace_local_rotation_xyz_deg'],[35.,-60.,50.])
        for side in n.SIDES:
            R=np.array(result['actor_pose_capture_by_side'][side]['core_C_capture'])[:3,:3]
            np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12)
        p['occluder_controls']['depth_fraction']=[-1,2]
        with self.assertRaises(ValueError):n.propose(a,e,p,n.sample_uniform(3,99),c,anchor_ref=ref)


if __name__=='__main__':unittest.main()
