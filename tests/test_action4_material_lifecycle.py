import copy
import unittest
from dataclasses import replace, FrozenInstanceError
from model.action4_phase2_tracking_contracts import AnchorKey, AnchorRegistry
from model.action4_material_lifecycle import (LifecyclePolicy, PredictedMaterialObservation,
    lifecycles_from_seed_packet, replace_binding)


def packet():
    # Exact field names emitted by seed_material_candidates. Values are a unit
    # fixture, not an actual A0 or CoTracker result.
    return {'schema':'action4.provisional_material_seeds.v1','query_role':'material_candidates',
        'synthetic_fixture_allowed':False,'time_identity_verified':True,'geometry_accepted':False,
        'mesh_binding_identity':'mesh-v1','track_id':'left','track_generation':2,'anchor_generation':0,
        'seeds':[{'key':{'track_id':'left','anchor_id':'image:b0:surface112:15','generation':0},
          'track_generation':2,'query_frame_id':10,'query_pts':100,'query_uv_native':[5.,6.],
          'X_cad_m':[.01,.02,.03],'mesh_triangle_id':7,'spatial_grid_id':1,'surface_grid_index':15,
          'XYZ_source':'A0_estimated_surface_nearest_fixed_mesh','binding_quality':'prediction_consistency_only',
          'binding_identity':'mesh-v1','lifecycle_state':'provisional','geometry_accepted':False}]}


def life(p=None, **kwargs):
    return next(iter(lifecycles_from_seed_packet(p or packet(),cad_diagonal_m=.1,**kwargs).values()))


def observation(state,frame,**changes):
    args=dict(key=state.key,track_generation=2,frame_id=frame,source_id=f'tracker:{frame}',
        raw_uv=(5.,6.),visibility=.95,confidence=.8,is_query=False,nonpadding=True,
        visibility_state='visible',sam_attribution='same_instance',sam_track_id='left',
        surface_xyz_m=(.01,.02,.03),surface_triangle_id=7,surface_supported=True,
        exclusions=tuple((k,'clear') for k in ('reflection','screen','strap','mount')),
        exclusion_source_id=f'predicted-exclusions:{frame}',surface_source_id=f'surface:{frame}',sam_source_id=f'sam:{frame}')
    args.update(changes)
    return PredictedMaterialObservation(**args)


class MaterialLifecycleTests(unittest.TestCase):
    def test_selector_shape_composes_with_existing_registry(self):
        s=life()
        for f in (11,12):
            s=s.observe(observation(s,f));self.assertEqual(s.state,'provisional')
        s=s.observe(observation(s,13));self.assertEqual(s.state,'active_for_geometry')
        self.assertEqual(s.confirmed_frames,(11,12,13));self.assertTrue(s.current_geometry_eligible)
        registry=AnchorRegistry([s.qualified_binding(frame_id=s.last_observation_frame_id)]);self.assertEqual(registry[s.key].xyz_m,(.01,.02,.03))
        with self.assertRaises(FrozenInstanceError):s.binding.xyz_m=(0.,0.,0.)

    def test_distinct_duplicate_idempotent_conflict_rejected(self):
        s=life();o=observation(s,11);s=s.observe(o)
        self.assertIs(s.observe(o),s);self.assertEqual(s.confirmed_frames,(11,))
        with self.assertRaisesRegex(ValueError,'Contradictory'):s.observe(replace(o,confidence=.7))

    def test_original_and_carried_queries_never_confirm(self):
        s=life()
        s=s.observe(observation(s,10,is_query=False));self.assertEqual(s.reason,'query_frame')
        s=s.observe(observation(s,11,is_query=True));self.assertEqual(s.confirmed_frames,())
        with self.assertRaises(ValueError):s.qualified_binding(frame_id=s.last_observation_frame_id)

    def test_unknown_and_occluded_distinct(self):
        for state in ('unknown','occluded'):
            s=life();s=s.observe(observation(s,11,visibility_state=state))
            self.assertEqual(s.reason,state);self.assertFalse(s.current_geometry_eligible)

    def test_continuous_and_coordinate_gates(self):
        cases=[({'raw_uv':(float('nan'),5.)},'nonfinite_track'),({'nonpadding':False},'padding'),
               ({'visibility':float('nan')},'invalid_continuous_scores'),({'confidence':1.01},'invalid_continuous_scores'),
               ({'visibility':.899},'low_continuous_scores'),({'confidence':.599},'low_continuous_scores')]
        for args,reason in cases:
            s=life().observe(observation(life(),11,**args));self.assertEqual(s.reason,reason);self.assertEqual(s.confirmed_frames,())
        with self.assertRaises(ValueError):observation(life(),11,visibility=True)

    def test_all_exclusion_categories_unknown_or_excluded(self):
        for category in ('reflection','screen','strap','mount'):
            for status in ('unknown','excluded'):
                s=life();e=tuple((k,status if k==category else 'clear') for k in ('reflection','screen','strap','mount'))
                s=s.observe(observation(s,11,exclusions=e))
                self.assertEqual(s.reason,'exclusion_unknown' if status=='unknown' else 'excluded_material')
                self.assertEqual(s.state,'provisional' if status=='unknown' else 'rejected')
        with self.assertRaises(ValueError):observation(life(),11,exclusions=(('reflection','clear'),))
        with self.assertRaises(ValueError):observation(life(),11,exclusion_source_id='')

    def test_instance_and_surface_qualification(self):
        cases=[({'sam_attribution':'unknown'},'sam_unknown'),({'sam_attribution':'other_instance'},'sam_other_instance'),
            ({'sam_track_id':'right'},'sam_identity_mismatch'),({'surface_supported':False},'surface_unknown'),
            ({'surface_triangle_id':None},'surface_unknown'),({'surface_xyz_m':None},'surface_unknown'),
            ({'surface_xyz_m':(float('inf'),0.,0.)},'surface_nonfinite'),({'surface_triangle_id':8},'surface_face_incompatible'),
            ({'surface_xyz_m':(.012001,.02,.03)},'surface_binding_distance')]
        for args,reason in cases:
            s=life();s=s.observe(observation(s,11,**args));self.assertEqual(s.reason,reason);self.assertEqual(s.confirmed_frames,())
        s=life();s=s.observe(observation(s,11,surface_xyz_m=(.01199,.02,.03)));self.assertEqual(s.confirmed_frames,(11,))

    def test_face_compatibility_requires_explicit_source(self):
        s=life(face_compatibility={7:{'triangle_ids':[7,8],'source_id':'frozen-CAD-coplanar-v1'}})
        s=s.observe(observation(s,11,surface_triangle_id=8));self.assertEqual(s.confirmed_frames,(11,))
        with self.assertRaises(ValueError):life(face_compatibility={7:{'triangle_ids':[8],'source_id':'e'}})
        with self.assertRaises(ValueError):life(face_compatibility={7:{'triangle_ids':[7,8],'source_id':''}})

    def test_identity_loss_or_generation_change_retires(self):
        for args,reason in (({'track_state':'lost'},'identity_lost'),({'track_generation':3},'track_generation_changed')):
            s=life();s=s.observe(observation(s,11,**args));self.assertEqual(s.state,'retired');self.assertEqual(s.reason,reason)
            with self.assertRaises(ValueError):s.observe(observation(s,12))

    def test_fixed_xyz_rebinding_requires_new_generation(self):
        p=packet();s=life(p);s.assert_seed_unchanged(p['seeds'][0])
        p['seeds'][0]['X_cad_m'][0]=.011
        with self.assertRaises(ValueError):s.assert_seed_unchanged(p['seeds'][0])
        with self.assertRaises(ValueError):replace_binding(s,life(p))
        p['anchor_generation']=1;p['seeds'][0]['key']['generation']=1
        old,new=replace_binding(s,life(p));self.assertEqual(old.state,'retired');self.assertEqual(new.state,'provisional')
        self.assertEqual(s.binding.xyz_m,(.01,.02,.03));self.assertEqual(new.binding.xyz_m,(.011,.02,.03))
        with self.assertRaises(ValueError):s.observe(observation(new,11))

    def test_no_future_solver_or_gt_or_fixture_admission(self):
        for origin in ('GT','PnP','solver','manual'):
            with self.assertRaises(ValueError):observation(life(),11,origin=origin)
        for field,value in (('query_role','engineering_input_not_material_seed'),('time_identity_verified',False),('synthetic_fixture_allowed',True),('geometry_accepted',True)):
            p=packet();p[field]=value
            with self.assertRaises(ValueError):life(p)

    def test_active_history_does_not_admit_bad_current_frame(self):
        s=life()
        for f in (11,12,13):s=s.observe(observation(s,f))
        s=s.observe(observation(s,14,visibility_state='unknown'))
        self.assertEqual(s.state,'active_for_geometry');self.assertFalse(s.current_geometry_eligible)
        with self.assertRaises(ValueError):s.qualified_binding(frame_id=s.last_observation_frame_id)
        s=s.observe(observation(s,15));self.assertTrue(s.current_geometry_eligible)
        excluded=tuple((k,'excluded' if k=='screen' else 'clear') for k in ('reflection','screen','strap','mount'))
        s=s.observe(observation(s,16,exclusions=excluded,confidence=.1));self.assertEqual(s.state,'rejected')

    def test_native_numpy_arrays_are_directly_consumable(self):
        import numpy as np
        s=life();o=observation(s,11,raw_uv=np.array([5,6],np.float32),visibility=np.float32(.95),
            confidence=np.float32(.8),surface_xyz_m=np.array([.01,.02,.03],np.float32))
        s=s.observe(o);self.assertEqual(s.confirmed_frames,(11,))
        self.assertIs(s.observe(o),s)

    def test_explicit_geometry_frame_binding_and_duplicate_does_not_move_it(self):
        p=packet();p['seeds'][0]['query_frame_id']=0
        s=life(p)
        for f in (11,12,13):s=s.observe(observation(s,f))
        s=s.observe(observation(s,20,visibility_state='unknown'))
        self.assertEqual(s.last_observation_frame_id,20)
        earlier=observation(s,10)
        s=s.observe(earlier)
        self.assertEqual(s.last_observation_frame_id,10)
        with self.assertRaisesRegex(ValueError,'Geometry frame'):
            s.qualified_binding(frame_id=20)
        self.assertEqual(s.qualified_binding(frame_id=10),s.binding)
        s=s.observe(observation(s,21))
        s=s.observe(earlier)
        self.assertEqual(s.last_observation_frame_id,21)
        with self.assertRaises(ValueError):s.qualified_binding(frame_id=10)
        self.assertEqual(s.qualified_binding(frame_id=21),s.binding)
        with self.assertRaises(TypeError):s.qualified_binding()

    def test_policy_and_duplicate_seed_guards(self):
        with self.assertRaises(ValueError):LifecyclePolicy(min_nonquery_frames=2)
        with self.assertRaises(ValueError):LifecyclePolicy(distance_fraction=.0201)
        p=packet();p['seeds'].append(copy.deepcopy(p['seeds'][0]))
        with self.assertRaises(ValueError):life(p)
        p=packet();p['seeds'][0]['XYZ_source']='GT'
        with self.assertRaises(ValueError):life(p)
        p=packet();p['seeds']=[]
        self.assertEqual(lifecycles_from_seed_packet(p,cad_diagonal_m=.1),{})

if __name__=='__main__':unittest.main()
