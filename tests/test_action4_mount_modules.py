"""Pure prepare/anchor/zone invariants. No Blender build or optimizer."""
import unittest,copy,json,sys
from pathlib import Path
import numpy as np
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import action4_mount_modules as m
CONFIG='configs/data/action4_mount_modules_v001.json'
class MountPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=m.load_config(CONFIG);cls.r=m.routes(cls.c)
    def test_original_radii_dimensions_and_reference_reused(self):
        source=m.read(self.c['sources']['source_contract']['path']);p=source['config']['initial_priors'];r=self.r
        self.assertEqual(r['link_radius_m'],p['link_radius_m']);self.assertEqual(r['cable_radius_m'],p['cable_radius_m']);self.assertEqual(r['ball_radius_m'],p['ball_radius_m'])
        for k,v in r['source_dimensions_m'].items():self.assertEqual(v,p[k])
        np.testing.assert_array_equal(np.array(r['old_brace_to_new_brace'])[:3,3],[0,-.003,0])
    def test_link_endpoint_binding_and_approach_on_CAD_negativeZ(self):
        r=self.r;A=np.array(r['A_new_brace_from_CAD']);link=np.array(r['derived_link_path_m']);q=np.array(r['quick_release_center_CAD_m'])
        np.testing.assert_array_equal(link[0],r['band_anchor_newbrace_m']);np.testing.assert_array_equal(link[1],r['ball_center_newbrace_m']);np.testing.assert_array_equal(link[-1],m.transform(A,q))
        local=m.transform(np.linalg.inv(A),link[-2:]);np.testing.assert_allclose(local[:,0:2],np.zeros((2,2)),atol=1e-14)
        self.assertLess(local[0,2],q[2]-r['source_dimensions_m']['quick_release_dimensions_m'][2]/2)
    def test_cable_exit_and_source_rail_magnitudes(self):
        r=self.r;new=np.array(r['derived_cable_path_m']);old=np.array(r['source_cable_path_reexpressed_m']);A=np.array(r['A_new_brace_from_CAD'])
        np.testing.assert_array_equal(new[0],m.transform(A,r['USB_exit_CAD_m']));np.testing.assert_array_equal(new[-4:,1:],old[-4:,1:]);np.testing.assert_array_equal(abs(new[-4:,0]),abs(old[-4:,0]))
        self.assertTrue(np.all(np.sign(new[-4:,0])==np.sign(new[0,0])));self.assertLess(m.transform(np.linalg.inv(A),new[1])[2],r['USB_exit_CAD_m'][2])
    def test_no_stale_bridge_endpoint_reuse(self):
        self.assertGreater(self.r['original_link_endpoint_gap_to_new_anchor_m'],.001);self.assertGreater(self.r['original_cable_start_gap_to_new_anchor_m'],.001)
    def test_proper_no_free_scale_or_reflection(self):
        for T in [np.diag([1.1,1,1,1]),np.diag([-1,1,1,1]),np.full((4,4),np.nan)]:
            with self.assertRaises(ValueError):m.rigid(T)
        A=m.rigid(self.r['A_new_brace_from_CAD']);C=np.array(m.read(self.c['sources']['joint_result']['path'])['C_per_frame']['960']);B=C@np.linalg.inv(A);np.testing.assert_allclose(B@A,C,rtol=0,atol=1e-14)
    def test_nominal_joint_candidate_not_acceptance(self):
        v=np.array([[0,0,0],[.1,0,0],[0,.1,0]]);f=np.array([[0,1,2]]);z={'module_to_zone':np.eye(4).tolist(),'min_m':[-.2]*3,'max_m':[.2]*3}
        d=m.classify_pairs(v,f,v,f,[(0,0)],z);self.assertEqual(d['within_nominal_joint_zone_candidates'],[[0,0]]);self.assertFalse(d['mechanical_acceptance'])
    def test_outside_zone_and_unlisted_pair_not_exempt(self):
        v=np.array([[1,0,0],[1.1,0,0],[1,.1,0]]);f=np.array([[0,1,2]]);z={'module_to_zone':np.eye(4).tolist(),'min_m':[-.2]*3,'max_m':[.2]*3}
        for zone in [z,None]:self.assertEqual(m.classify_pairs(v,f,v,f,[(0,0)],zone)['outside_or_unresolved_pairs'],[[0,0]])
        self.assertFalse(m.classify_pairs(v,f,v,f,[],None)['mechanical_acceptance'])
    def test_zone_frame_transform_is_applied(self):
        v=np.array([[1,0,0],[1.1,0,0],[1,.1,0]]);f=np.array([[0,1,2]]);M=np.eye(4);M[0,3]=-1;z={'module_to_zone':M.tolist(),'min_m':[-.2]*3,'max_m':[.2]*3}
        self.assertEqual(m.classify_pairs(v,f,v,f,[(0,0)],z)['within_nominal_joint_zone_candidates'],[[0,0]])
    def test_config_scale_route_generation_overrides_rejected(self):
        original=m.read
        for change in ['route','generation','identity']:
            bad=copy.deepcopy(self.c)
            if change=='route':bad['routing']['quick_approach_clearance_m']=.1
            elif change=='generation':bad['dataset_generation_allowed']=True
            else:bad['sources']['joint_result']['sha256']='0'*64
            with patch.object(m,'read',side_effect=lambda p:bad if p=='bad_fixture' else original(p)):
                with self.assertRaises(ValueError):m.load_config('bad_fixture')
    def test_unapproved_build_stops_before_any_attempt_or_Blender(self):
        original=m.read
        with patch.object(m,'read',side_effect=lambda p:{'status':'NOT_APPROVED'} if p=='no_approval_fixture' else original(p)),patch.object(m,'_build_once',side_effect=AssertionError('build forbidden')):
            with self.assertRaises(ValueError):m.build(CONFIG,'no_approval_fixture')
        self.assertFalse((m.OUTPUT/'actual_build_v001').exists())
    def test_approved_receipt_cannot_switch_requested_config(self):
        ap={'status':'APPROVED_SINGLE_CPU_RIGHT_MOUNT_BUILD','preparation_handoff_sha256':'bound','spec_review_pass':True,'spec_review_path':'spec','spec_review_sha256':'bound','root_review_pass':True,'root_review_path':'root','root_review_sha256':'bound'}
        handoff={'files':{},'config':{'path':'canonical_config'}};wrong=copy.deepcopy(self.c);wrong['sources']['joint_result']['path']='different_result'
        with patch.object(m,'sha',return_value='bound'),patch.object(m,'read',side_effect=lambda p:ap if p=='approval_fixture' else handoff),patch.object(m,'load_config',return_value=self.c):
            with self.assertRaises(ValueError):m._approval(wrong,'approval_fixture')
    def test_collision_skin_uses_reviewedS_and_matches_actual_context(self):
        V,F,meta=m.collision_skin_context(self.c);lib=np.load(m.path(self.c['sources']['wrist_geometry']['path']),allow_pickle=False);rel=m.read(self.c['sources']['right_relation_config']['path']);S=np.array(rel['S_skin_relative_to_brace'])
        np.testing.assert_array_equal(V,m.transform(S,lib['brace_vertices_m']));np.testing.assert_array_equal(F,lib['faces']);self.assertLess(meta['maximum_previous_actual_local_difference_m'],1e-6);self.assertGreater(meta['identity_substitution_max_difference_m'],.008)
        self.assertFalse(meta['wearing_alignment_accepted']);self.assertFalse(meta['S_adjusted_by_this_fix'])
    def test_identity_context_substitution_rejected(self):
        lib=np.load(m.path(self.c['sources']['wrist_geometry']['path']),allow_pickle=False)
        with self.assertRaises(ValueError):m.collision_skin_context(self.c,(lib['brace_vertices_m'],lib['faces']))
    def test_other_S_not_accepted_from_same_relation_reference(self):
        original=m.read;relation_path=self.c['sources']['right_relation_config']['path'];bad=copy.deepcopy(original(relation_path));bad['S_skin_relative_to_brace']=np.eye(4).tolist()
        with patch.object(m,'read',side_effect=lambda p:bad if p==relation_path else original(p)):
            with self.assertRaises(ValueError):m.collision_skin_context(self.c)
    def test_routes_exactly_unchanged_by_context_fix(self):
        old=m.read(m.OUTPUT/'before_skin_context_fix_v001/preparation.json')
        self.assertEqual(m.routes(self.c),old['routes'])
if __name__=='__main__':unittest.main(verbosity=2)
