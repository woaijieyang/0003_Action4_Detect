"""Pure CPU history contracts; no model execution, I/O or output publication."""
import unittest
import numpy as np

from model.action4_observation_history import AcceptedObservationHistory,ObservationKey

SPACE="raw_video_native_integer_centers"


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.history=AcceptedObservationHistory(max_horizon_seconds=2.,max_gap_seconds=3.)
        self.key=ObservationKey("body-A",0,"corner",0,0)

    def put(self,frame,pts,position,**kwargs):
        options=dict(frame_id=frame,pts=pts,time_base=(1,100),pixel_space_id=SPACE,
            source_kind="image_keypoint",evidence_id=f"image:{frame}",accepted_current_image=True)
        options.update(kwargs)
        return self.history.update(self.key,position,**options)

    def prior(self,frame,pts,**kwargs):
        options=dict(frame_id=frame,pts=pts,time_base=(1,100),pixel_space_id=SPACE)
        options.update(kwargs)
        return self.history.prior(self.key,**options)

    def test_actual_nonuniform_pts_and_exact_rational_delta(self):
        self.put(1,100,[0.,0.]);self.put(2,130,[3.,6.])
        result=self.prior(3,200)
        np.testing.assert_allclose(result["velocity_px_per_second"],[10.,20.])
        np.testing.assert_allclose(result["position"],[10.,20.])
        self.assertEqual(result["horizon_seconds"],.7)
        self.assertTrue(result["prediction_only"]);self.assertFalse(result["publishable"])
        h=AcceptedObservationHistory(max_horizon_seconds=2,max_gap_seconds=3)
        for frame,pts,point in ((0,10**18,[0.,0.]),(1,10**18+1,[1.,2.])):
            h.update(self.key,point,frame_id=frame,pts=pts,time_base=(1,100),pixel_space_id=SPACE,
                source_kind="image_keypoint",evidence_id=str(frame),accepted_current_image=True)
        result=h.prior(self.key,frame_id=2,pts=10**18+2,time_base=(1,100),pixel_space_id=SPACE)
        self.assertEqual(result["position"],[2.,4.])

    def test_only_last_five_actual_accepted_observations(self):
        for i in range(8):self.put(i,i*10,[float(i),0.])
        rows=self.history.observations(self.key)
        self.assertEqual([r["frame_id"] for r in rows],[3,4,5,6,7])
        rows[0]["position"][0]=999
        self.assertEqual(self.history.observations(self.key)[0]["position"][0],3.)

    def test_predictions_rejected_missing_and_centroid_never_enter(self):
        self.put(0,0,[0.,0.]);self.put(1,10,[1.,0.])
        for index,kind in enumerate(("predicted_only","rejected","missing","raw_mask_centroid"),2):
            result=self.put(index,index*10,None if kind=="missing" else [900.,900.],
                            source_kind=kind,accepted_current_image=False)
            self.assertFalse(result["stored"])
            self.assertEqual(len(self.history.observations(self.key)),2)
            with self.assertRaises(ValueError):
                self.put(index,index*10,[900.,900.],source_kind=kind,accepted_current_image=True)

    def test_current_fast_motion_is_not_rejected_or_smoothed(self):
        self.put(0,0,[0.,0.]);self.put(1,10,[1.,0.])
        result=self.put(2,20,[-90.,80.])
        self.assertTrue(result["stored"])
        self.assertEqual(result["prior"]["position"],[2.,0.])
        self.assertEqual(result["innovation"],[-92.,80.])
        self.assertEqual(result["measurement"],[-90.,80.])
        self.assertEqual(self.history.observations(self.key)[-1]["position"],[-90.,80.])

    def test_no_history_insufficient_horizon_and_gap_unknown(self):
        self.assertEqual(self.prior(0,0)["status"],"unknown")
        self.put(0,0,[0.,0.])
        self.assertEqual(self.prior(1,10)["reason"],"insufficient_history")
        self.put(1,10,[1.,0.])
        self.assertEqual(self.prior(2,220)["reason"],"maximum_horizon_exceeded")
        result=self.put(2,500,[9.,8.])
        self.assertEqual(result["history_reset"],"observation_gap")
        self.assertEqual(len(self.history.observations(self.key)),1)
        self.assertEqual(self.prior(3,510)["reason"],"insufficient_history")

    def test_generation_changes_unknown_and_old_body_invalidates_other_nodes(self):
        self.put(0,0,[0.,0.]);self.put(1,10,[1.,0.])
        new=ObservationKey("body-A",0,"corner",0,1)
        self.assertEqual(self.history.prior(new,frame_id=2,pts=20,time_base=(1,100),pixel_space_id=SPACE)["status"],"unknown")
        self.history.update(ObservationKey("body-A",1,"body_center",0,0),[3.,4.],
            frame_id=2,pts=20,time_base=(1,100),pixel_space_id=SPACE,source_kind="image_detection_center",
            evidence_id="new-body",accepted_current_image=True)
        self.assertEqual(self.prior(3,30)["status"],"unknown")
        with self.assertRaisesRegex(ValueError,"Stale body"):self.put(3,30,[3.,0.])

    def test_same_observation_idempotent_same_frame_conflict_and_time_domain_rejected(self):
        self.put(0,0,[1.,2.])
        result=self.put(0,0,[1.,2.]);self.assertTrue(result["idempotent"])
        self.assertEqual(len(self.history.observations(self.key)),1)
        for changes in (dict(position=[2.,2.]),dict(pts=1),dict(evidence_id="other")):
            data=dict(frame=0,pts=0,position=[1.,2.]);data.update(changes)
            with self.assertRaises(ValueError):self.put(**data)
        for kwargs in (dict(time_base=(1,1000)),dict(pixel_space_id="native_integer_centers"),
                       dict(position=[[1.,2.],[3.,4.]]),dict(pts=-1),dict(evidence_id="image:0")):
            data=dict(frame=1,pts=10,position=[2.,3.]);data.update(kwargs)
            with self.assertRaises(ValueError):self.put(**data)
        with self.assertRaises(ValueError):ObservationKey("body-A",0,"face",6,0)
        with self.assertRaises(ValueError):ObservationKey("body-A",0,"corner",True,0)
        other=ObservationKey("body-B",0,"corner",0,0)
        self.assertEqual(self.history.prior(other,frame_id=1,pts=10,time_base=(1,100),pixel_space_id=SPACE)["status"],"unknown")

    def test_rigid_pose_groups_and_quad_rotation_innovation_are_diagnostic(self):
        self.key=ObservationKey("body-A",0,"face",4,0)
        q=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        for i in (0,1):
            self.put(i,i*10,q,source_kind="image_rigid_pose_projection",evidence_group_id=f"body-pose:frame:{i}")
        rotated=q@np.array([[0.,-1.],[1.,0.]]).T
        result=self.put(2,20,rotated,source_kind="image_rigid_pose_projection",evidence_group_id="body-pose:frame:2")
        np.testing.assert_array_equal(result["innovation"],rotated-q)
        self.assertTrue(result["stored"])
        self.assertTrue(result["measurement_evidence"]["correlated"])
        self.assertFalse(result["measurement_evidence"]["physical_corner_observed"])
        for row in self.history.observations(self.key):
            self.assertTrue(row["correlated"])
            self.assertFalse(row["independent_ground_truth"])
            self.assertIn("body-pose:frame:",row["evidence_group_id"])
        with self.assertRaises(ValueError):
            self.put(3,30,q,source_kind="image_rigid_pose_projection")

    def test_invalid_measurement_and_numerical_overflow_unknown(self):
        for point in ([np.nan,0.],[],[1.,2.,3.]):
            with self.assertRaises(ValueError):self.put(0,0,point)
        self.put(0,0,[-1e308,0.]);self.put(1,10,[1e308,0.])
        result=self.prior(2,20)
        self.assertEqual(result["reason"],"nonfinite_extrapolation")
        self.assertIsNone(result["position"])
        for value in (0.,-1.,float("nan"),True):
            with self.assertRaises(ValueError):
                AcceptedObservationHistory(max_horizon_seconds=value,max_gap_seconds=3.)



    def test_eight_rigid_projected_corners_preserve_shared_pose_groups(self):
        for frame in (0,1):
            for corner_id in range(8):
                key=ObservationKey("body-A",0,"corner",corner_id,0)
                self.history.update(key,[float(corner_id+frame),2.],
                    frame_id=frame,pts=frame*10,time_base=(1,100),pixel_space_id=SPACE,
                    source_kind="image_rigid_pose_projection",
                    evidence_id=f"pose:{frame}:corner:{corner_id}",
                    evidence_group_id=f"body-A:pose:{frame}",accepted_current_image=True)
        for corner_id in range(8):
            key=ObservationKey("body-A",0,"corner",corner_id,0)
            prior=self.history.prior(key,frame_id=2,pts=20,time_base=(1,100),pixel_space_id=SPACE)
            self.assertEqual(prior["evidence_group_ids"],["body-A:pose:0","body-A:pose:1"])
            self.assertTrue(prior["correlated"])
            self.assertFalse(prior["publishable"])
            self.assertFalse(prior["independent_ground_truth"])

    def test_short_gap_limit_is_distinct_from_horizon(self):
        self.history=AcceptedObservationHistory(max_horizon_seconds=10.,max_gap_seconds=.5)
        self.put(0,0,[0.,0.]);self.put(1,10,[1.,0.])
        self.assertEqual(self.prior(2,70)["reason"],"observation_gap_exceeded")



    def test_first_three_observations_change_prior_with_identical_last_two(self):
        results=[]
        for xs in ([0.,0.,0.,10.,11.],[-20.,-10.,0.,10.,11.]):
            self.history=AcceptedObservationHistory(max_horizon_seconds=2.,max_gap_seconds=3.)
            for i,x in enumerate(xs):self.put(i,i*10,[x,0.])
            results.append(self.prior(5,50))
        self.assertEqual(results[0]["position"],[11.5,0.])
        self.assertEqual(results[1]["position"],[21.,0.])
        for result in results:
            self.assertEqual(result["used_observation_count"],5)
            self.assertEqual(result["used_segment_count"],4)
            self.assertEqual([x["frame_id"] for x in result["source_observations"]],[0,1,2,3,4])
            self.assertEqual(result["extrapolation_anchor_frame_id"],4)

    def test_single_reverse_jump_does_not_dominate_median_or_modify_latest_observation(self):
        for i,x in enumerate([0.,1.,2.,3.,-7.]):self.put(i,i*100,[x,0.])
        result=self.prior(5,500)
        self.assertEqual(result["velocity_px_per_second"],[1.,0.])
        self.assertEqual(result["position"],[-6.,0.])
        self.assertEqual(self.history.observations(self.key)[-1]["position"],[-7.,0.])
        self.assertEqual(result["velocity_segments"][-1]["velocity_px_per_second"],[-10.,0.])
        self.assertFalse(result["publishable"])

    def test_all_five_irregular_pts_segments_use_actual_dt(self):
        for frame,pts in enumerate([0,30,90,100,180]):
            self.put(frame,pts,[2*pts/100,3*pts/100])
        result=self.prior(5,210)
        np.testing.assert_allclose(result["velocity_px_per_second"],[2.,3.])
        np.testing.assert_allclose(result["position"],[4.2,6.3])
        self.assertEqual([s["dt_seconds"] for s in result["velocity_segments"]],[.3,.6,.1,.8])
        self.assertEqual([(s["from_pts"],s["to_pts"]) for s in result["velocity_segments"]],
                         [(0,30),(30,90),(90,100),(100,180)])

    def test_one_bad_older_segment_invalidates_entire_prior(self):
        for i,x in enumerate([-1e308,1e308,0.,1.,2.]):self.put(i,i*10,[x,0.])
        result=self.prior(5,50)
        self.assertEqual(result["status"],"unknown")
        self.assertEqual(result["reason"],"nonfinite_extrapolation")
        self.assertEqual(result["accepted_observation_count"],5)
        self.assertEqual(result["used_observation_count"],0)
        self.assertIsNone(result["position"])



    def test_missing_or_prediction_watermark_prevents_past_accepted_insertion(self):
        for kind in ("missing","predicted_only"):
            self.history=AcceptedObservationHistory(max_horizon_seconds=20.,max_gap_seconds=30.)
            self.put(4,400,[4.,0.]);self.put(5,500,[5.,0.])
            self.put(10,1000,None if kind=="missing" else [10.,0.],
                     source_kind=kind,accepted_current_image=False)
            before=self.history.observations(self.key)
            with self.assertRaisesRegex(ValueError,"watermark"):
                self.put(8,800,[8.,0.])
            self.assertEqual(self.history.observations(self.key),before)
            # Exact retained accepted replay is a no-op, not a new insertion.
            self.assertTrue(self.put(5,500,[5.,0.])["idempotent"])
            with self.assertRaisesRegex(ValueError,"watermark"):
                self.put(9,900,[9.,0.])

    def test_nonaccepted_generation_events_retire_old_body_and_node_identity(self):
        self.put(4,400,[4.,0.]);self.put(5,500,[5.,0.])
        new=ObservationKey("body-A",1,"corner",0,0)
        self.history.update(new,None,frame_id=10,pts=1000,time_base=(1,100),pixel_space_id=SPACE,
            source_kind="missing",evidence_id="missing-new-body",accepted_current_image=False)
        self.assertEqual(self.prior(11,1100)["status"],"unknown")
        with self.assertRaisesRegex(ValueError,"Stale body"):
            self.put(11,1100,[11.,0.])
        self.assertEqual(self.history.observations(new),[])
        self.assertEqual(self.history.prior(new,frame_id=11,pts=1100,time_base=(1,100),
                                           pixel_space_id=SPACE)["reason"],"no_accepted_history")
        newer=ObservationKey("body-A",1,"corner",0,1)
        self.history.update(newer,None,frame_id=11,pts=1100,time_base=(1,100),pixel_space_id=SPACE,
            source_kind="missing",evidence_id="missing-new-node",accepted_current_image=False)
        with self.assertRaisesRegex(ValueError,"Stale body/node"):
            self.history.update(new,[1.,2.],frame_id=12,pts=1200,time_base=(1,100),pixel_space_id=SPACE,
                source_kind="image_keypoint",evidence_id="old-node",accepted_current_image=True)

    def test_same_frame_nonaccepted_to_accepted_upgrade_once(self):
        self.put(0,0,[0.,0.]);self.put(1,10,[1.,0.])
        result=self.put(2,20,[999.,999.],source_kind="predicted_only",accepted_current_image=False)
        self.assertFalse(result["stored"])
        actual=self.put(2,20,[3.,4.])
        self.assertTrue(actual["stored"]);self.assertTrue(actual["same_frame_upgrade"])
        self.assertEqual(len(self.history.observations(self.key)),3)
        self.assertEqual(actual["innovation"],[1.,4.])
        self.assertTrue(self.put(2,20,[3.,4.])["idempotent"])
        with self.assertRaises(ValueError):self.put(2,20,[5.,6.])
        with self.assertRaises(ValueError):
            self.put(2,20,None,source_kind="missing",accepted_current_image=False)

    def test_processed_domain_pts_shape_and_independent_node_order(self):
        self.put(1,10,None,source_kind="missing",accepted_current_image=False)
        for kwargs in (dict(pts=11),dict(time_base=(1,1000)),dict(pixel_space_id="native_integer_centers")):
            data=dict(frame=1,pts=10,position=[1.,2.]);data.update(kwargs)
            with self.assertRaises(ValueError):self.put(**data)
        other=ObservationKey("body-A",0,"corner",1,0)
        for key in (other,self.key):
            self.history.update(key,[1.,2.],frame_id=1,pts=10,time_base=(1,100),pixel_space_id=SPACE,
                source_kind="image_keypoint",evidence_id="same-frame-different-node:"+str(key.node_id),
                accepted_current_image=True)
            self.assertEqual(len(self.history.observations(key)),1)
        # Querying a far-future prior must not advance processing time.
        self.history.prior(self.key,frame_id=100,pts=1000,time_base=(1,100),pixel_space_id=SPACE)
        self.assertTrue(self.put(2,20,[2.,3.])["stored"])
        with self.assertRaises(ValueError):
            self.put(3,30,[[1.,2.]],source_kind="predicted_only",accepted_current_image=False)


if __name__=="__main__":
    unittest.main()
